from __future__ import annotations

import torch
import torch.nn as nn


class ChannelGraphBandSelector(nn.Module):
    """Learnable automatic band selection with random channel-graph initialization."""

    def __init__(
        self,
        channels: int,
        keep_bands: int,
        graph_rank: int = 16,
        score_hidden: int = 16,
    ) -> None:
        super().__init__()
        if keep_bands <= 0:
            raise ValueError("keep_bands must be positive.")
        self.channels = channels
        self.keep_bands = min(keep_bands, channels)
        self.graph_rank = min(graph_rank, channels)

        self.left_factor = nn.Parameter(torch.randn(channels, self.graph_rank) * 0.02)
        self.right_factor = nn.Parameter(torch.randn(channels, self.graph_rank) * 0.02)
        self.band_bias = nn.Parameter(torch.zeros(channels))
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, 1),
        )

    def _normalized_graph(self) -> torch.Tensor:
        w = torch.relu(self.left_factor @ self.right_factor.transpose(0, 1))
        eye = torch.eye(self.channels, device=w.device, dtype=w.dtype)
        w = w * (1.0 - eye)
        w = w / (w.amax().clamp_min(1e-6))
        return w

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if x1.shape != x2.shape:
            raise ValueError(f"x1/x2 shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}")
        if x1.ndim != 3:
            raise ValueError("band selector expects [batch, channels, patch_pixels].")
        _, channels, patch_pixels = x1.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} bands, got {channels}.")

        diff_mean = (x2 - x1).abs().mean(dim=-1)
        diff_std = (x2 - x1).std(dim=-1)
        center_diff = (x2[:, :, patch_pixels // 2] - x1[:, :, patch_pixels // 2]).abs()
        stats = torch.stack([diff_mean, diff_std, center_diff], dim=-1)

        data_logits = self.score_mlp(stats).squeeze(-1)
        graph = self._normalized_graph()
        graph_logits = graph.mean(dim=-1).unsqueeze(0) + self.band_bias.unsqueeze(0)
        logits = data_logits + graph_logits

        band_prob = logits.softmax(dim=1)
        topk = torch.topk(logits, k=self.keep_bands, dim=1, largest=True, sorted=False).indices
        selected_idx = torch.sort(topk, dim=1).values
        gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, patch_pixels)
        gate = 1.0 + torch.gather(band_prob, dim=1, index=selected_idx).unsqueeze(-1) * channels
        x1_sel = torch.gather(x1, dim=1, index=gather_idx) * gate
        x2_sel = torch.gather(x2, dim=1, index=gather_idx) * gate
        info = {
            "indices": selected_idx,
            "scores": logits,
            "probabilities": band_prob,
            "graph": graph,
        }
        return x1_sel, x2_sel, info


class SpectralSpatialTransformer(nn.Module):
    def __init__(
        self,
        input_bands: int,
        patch_pixels: int,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_pixels, dim)
        self.band_embedding = nn.Embedding(input_bands, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, band_indices: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_to_embedding(x)
        tokens = tokens + self.band_embedding(band_indices)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.norm(self.encoder(tokens))


class ResidualPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x)) + x


class PreNormOnly(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectedAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(b, n, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        ]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class ProjectedCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x_qkv: torch.Tensor) -> torch.Tensor:
        b, n, _ = x_qkv.shape
        q = self.to_q(x_qkv[:, :1]).view(b, 1, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x_qkv).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x_qkv).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, 1, self.heads * self.dim_head)
        return self.to_out(out)


class SSTBranch(nn.Module):
    """SSTFormer branch: band-token transformer followed by spectral embedding transformer."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        b_dim: int,
        b_depth: int,
        b_heads: int,
        b_dim_head: int,
        b_mlp_dim: int,
        num_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualPreNorm(dim, ProjectedAttention(dim, heads, dim_head, dropout)),
                        ResidualPreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.channels_to_embedding = nn.Linear(num_tokens, b_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, b_dim) * 0.02)
        self.b_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualPreNorm(b_dim, ProjectedAttention(b_dim, b_heads, b_dim_head, dropout)),
                        ResidualPreNorm(b_dim, FeedForward(b_dim, b_mlp_dim, dropout)),
                    ]
                )
                for _ in range(b_depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        x = self.channels_to_embedding(x.transpose(1, 2))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for attn, ff in self.b_layers:
            x = attn(x)
            x = ff(x)
        return x


class SSTTemporalCrossFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 3,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [PreNormOnly(dim, ProjectedCrossAttention(dim, heads, dim_head, dropout)) for _ in range(depth)]
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out1 = x1[:, :1]
        out2 = x2[:, :1]
        for cross_attn in self.layers:
            x1_cls, x1_tokens = x1[:, :1], x1[:, 1:]
            x2_cls, x2_tokens = x2[:, :1], x2[:, 1:]
            out1 = x1_cls + cross_attn(torch.cat([x1_cls, x2_tokens], dim=1))
            x1 = torch.cat([out1, x1_tokens], dim=1)
            out2 = x2_cls + cross_attn(torch.cat([x2_cls, x1], dim=1))
            x2 = torch.cat([out2, x2_tokens], dim=1)
        return out1, out2


class SSTFormerReference(nn.Module):
    """Faithful local implementation of the published SSTFormer classifier."""

    def __init__(
        self,
        input_bands: int,
        patch_size: int,
        num_classes: int = 2,
        dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 8,
        b_dim: int = 512,
        b_depth: int = 3,
        b_heads: int = 8,
        b_dim_head: int = 32,
        b_mlp_dim: int = 8,
        cross_depth: int = 3,
        dropout: float = 0.2,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        patch_pixels = patch_size * patch_size
        token_count = input_bands + 1
        self.patch_to_embedding = nn.Linear(patch_pixels, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_count, dim) * 0.02)
        self.cls_token_t1 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.branch = SSTBranch(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            b_dim=b_dim,
            b_depth=b_depth,
            b_heads=b_heads,
            b_dim_head=b_dim_head,
            b_mlp_dim=b_mlp_dim,
            num_tokens=token_count,
            dropout=dropout,
        )
        self.temporal_fusion = SSTTemporalCrossFusion(
            dim=b_dim,
            depth=cross_depth,
            heads=b_heads,
            dim_head=b_dim_head,
            dropout=0.0,
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(b_dim), nn.Linear(b_dim, num_classes))

    def _embed(self, x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        x = self.patch_to_embedding(x)
        cls = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : x.shape[1]]
        return self.dropout(x)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z1 = self.branch(self._embed(x1, self.cls_token_t1))
        z2 = self.branch(self._embed(x2, self.cls_token_t2))
        out1, out2 = self.temporal_fusion(z1, z2)
        logits = self.mlp_head((out1[:, 0] + out2[:, 0]))
        if return_info:
            return logits, {}
        return logits


class AdaptiveConditionInjector(nn.Module):
    """Lightweight ACor/ACI-style fusion with sample-level adaptive injection."""

    def __init__(self, dim: int, hidden_mult: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(dim, dim * hidden_mult)
        self.norm_base = nn.LayerNorm(dim)
        self.param_net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim * 3),
        )
        self.value_proj = nn.Linear(dim * 2, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, t1_tokens: torch.Tensor, t2_tokens: torch.Tensor) -> torch.Tensor:
        change = (t2_tokens - t1_tokens).abs()
        context = torch.cat([t1_tokens[:, 0], t2_tokens[:, 0], change.mean(dim=1)], dim=-1)
        params = self.param_net(context)
        gamma, beta, gate = params.chunk(3, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        gate = torch.sigmoid(gate).unsqueeze(1)
        base = self.value_proj(torch.cat([change, t1_tokens + t2_tokens], dim=-1))
        injected = self.norm_base(base) * (1.0 + gamma) + beta
        return self.out_norm(base + gate * injected)


class HSIChangeDetector(nn.Module):
    def __init__(
        self,
        input_bands: int,
        patch_size: int,
        keep_bands: int,
        num_classes: int = 2,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        selector_rank: int = 16,
    ) -> None:
        super().__init__()
        patch_pixels = patch_size * patch_size
        self.selector = ChannelGraphBandSelector(input_bands, keep_bands, graph_rank=selector_rank)
        self.encoder = SpectralSpatialTransformer(
            input_bands=input_bands,
            patch_pixels=patch_pixels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.aci = AdaptiveConditionInjector(dim=dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1_sel, x2_sel, selection = self.selector(x1, x2)
        h1 = self.encoder(x1_sel, selection["indices"])
        h2 = self.encoder(x2_sel, selection["indices"])
        fused = self.aci(h1, h2)
        cls_feature = fused[:, 0]
        band_feature = fused[:, 1:].mean(dim=1)
        logits = self.classifier(torch.cat([cls_feature, band_feature], dim=-1))
        if return_info:
            return logits, selection
        return logits


class SelectedSSTACIDetector(nn.Module):
    """Automatic band selection plus SSTFormer-style encoder and ACI-only fusion."""

    def __init__(
        self,
        input_bands: int,
        patch_size: int,
        keep_bands: int,
        num_classes: int = 2,
        dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 8,
        b_dim: int = 256,
        b_depth: int = 2,
        b_heads: int = 4,
        b_dim_head: int = 32,
        b_mlp_dim: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        selector_rank: int = 16,
    ) -> None:
        super().__init__()
        patch_pixels = patch_size * patch_size
        token_count = min(keep_bands, input_bands) + 1
        self.selector = ChannelGraphBandSelector(input_bands, keep_bands, graph_rank=selector_rank)
        self.patch_to_embedding = nn.Linear(patch_pixels, dim)
        self.band_embedding = nn.Embedding(input_bands, dim)
        self.cls_token_t1 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.branch = SSTBranch(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            b_dim=b_dim,
            b_depth=b_depth,
            b_heads=b_heads,
            b_dim_head=b_dim_head,
            b_mlp_dim=b_mlp_dim,
            num_tokens=token_count,
            dropout=dropout,
        )
        self.aci = AdaptiveConditionInjector(dim=b_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(b_dim * 2),
            nn.Linear(b_dim * 2, b_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(b_dim // 2, num_classes),
        )

    def _embed(self, x: torch.Tensor, band_indices: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        x = self.patch_to_embedding(x) + self.band_embedding(band_indices)
        cls = cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat([cls, x], dim=1))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x1_sel, x2_sel, selection = self.selector(x1, x2)
        z1 = self.branch(self._embed(x1_sel, selection["indices"], self.cls_token_t1))
        z2 = self.branch(self._embed(x2_sel, selection["indices"], self.cls_token_t2))
        fused = self.aci(z1, z2)
        logits = self.classifier(torch.cat([fused[:, 0], fused[:, 1:].mean(dim=1)], dim=-1))
        if return_info:
            return logits, selection
        return logits


class FixedBandSSTACIDetector(nn.Module):
    """Fixed automatic band subset with SSTFormer-style encoder and TAM fusion."""

    def __init__(
        self,
        input_bands: int,
        patch_size: int,
        selected_bands: torch.Tensor | list[int],
        num_classes: int = 2,
        dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 8,
        b_dim: int = 256,
        b_depth: int = 2,
        b_heads: int = 4,
        b_dim_head: int = 32,
        b_mlp_dim: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        patch_pixels = patch_size * patch_size
        bands = torch.as_tensor(selected_bands, dtype=torch.long)
        if bands.ndim != 1 or bands.numel() == 0:
            raise ValueError("selected_bands must be a non-empty 1D list/tensor.")
        if int(bands.min()) < 0 or int(bands.max()) >= input_bands:
            raise ValueError("selected_bands contains an index outside input_bands.")
        self.register_buffer("selected_bands", bands)
        token_count = int(bands.numel()) + 1
        self.patch_to_embedding = nn.Linear(patch_pixels, dim)
        self.band_embedding = nn.Embedding(input_bands, dim)
        self.cls_token_t1 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.branch = SSTBranch(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            b_dim=b_dim,
            b_depth=b_depth,
            b_heads=b_heads,
            b_dim_head=b_dim_head,
            b_mlp_dim=b_mlp_dim,
            num_tokens=token_count,
            dropout=dropout,
        )
        self.aci = AdaptiveConditionInjector(dim=b_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(b_dim * 2),
            nn.Linear(b_dim * 2, b_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(b_dim // 2, num_classes),
        )

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, dim=1, index=self.selected_bands)

    def _embed(self, x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        band_indices = self.selected_bands.unsqueeze(0).expand(x.shape[0], -1)
        x = self.patch_to_embedding(x) + self.band_embedding(band_indices)
        cls = cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat([cls, x], dim=1))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z1 = self.branch(self._embed(self._select(x1), self.cls_token_t1))
        z2 = self.branch(self._embed(self._select(x2), self.cls_token_t2))
        fused = self.aci(z1, z2)
        feature = torch.cat([fused[:, 0], fused[:, 1:].mean(dim=1)], dim=-1)
        logits = self.classifier(feature)
        if return_info:
            indices = self.selected_bands.unsqueeze(0).expand(x1.shape[0], -1)
            return logits, {"indices": indices}
        return logits
