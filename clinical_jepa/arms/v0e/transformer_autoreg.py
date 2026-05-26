from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TARGET_ENCODER_MODES = ("shared_sequence_encoder_stop_gradient", "fixed_mean_token")


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on optional torch
        raise ImportError("Transformer autoregression requires the optional torch dependency") from exc
    return torch, nn


@dataclass(frozen=True)
class TransformerAutoregConfig:
    vocab_size: int
    embedding_dim: int = 128
    max_horizons: int = 3
    encoder_layers: int = 2
    heads: int = 4
    max_len: int = 256
    ff_mult: int = 4
    dropout: float = 0.1
    use_layer_norm: bool = True
    predictor_hidden_mult: int = 2
    target_encoder_mode: str = "shared_sequence_encoder_stop_gradient"

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "TransformerAutoregConfig":
        arch = checkpoint.get("architecture") or {}
        target_encoder_mode = checkpoint.get("target_encoder_mode") or arch.get("target_encoder_mode")
        if target_encoder_mode is None:
            # Backward compatibility for 0aeda4a checkpoints, whose shared context/target
            # encoder could collapse under the governed BP006 primary run.
            target_encoder_mode = "shared_sequence_encoder_stop_gradient"
        return cls(
            vocab_size=int(checkpoint.get("vocab_size") or arch.get("vocab_size")),
            embedding_dim=int(checkpoint.get("embedding_dim") or checkpoint.get("dim") or arch.get("embedding_dim", 128)),
            max_horizons=int(checkpoint.get("max_horizons") or arch.get("max_horizons") or checkpoint.get("horizon_count_trained") or 3),
            encoder_layers=int(checkpoint.get("encoder_layers") or arch.get("encoder_layers", 2)),
            heads=int(checkpoint.get("heads") or arch.get("heads", 4)),
            max_len=int(checkpoint.get("max_len") or arch.get("max_len", 256)),
            ff_mult=int(checkpoint.get("ff_mult") or arch.get("ff_mult", 4)),
            dropout=float(checkpoint.get("dropout") if checkpoint.get("dropout") is not None else arch.get("dropout", 0.1)),
            use_layer_norm=bool(checkpoint.get("use_layer_norm") if checkpoint.get("use_layer_norm") is not None else arch.get("use_layer_norm", True)),
            predictor_hidden_mult=int(checkpoint.get("predictor_hidden_mult") or arch.get("predictor_hidden_mult", 2)),
            target_encoder_mode=str(target_encoder_mode),
        )

    def __post_init__(self) -> None:
        if self.target_encoder_mode not in TARGET_ENCODER_MODES:
            raise ValueError(f"target_encoder_mode must be one of {TARGET_ENCODER_MODES}; got {self.target_encoder_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "transformer-autoregressive-latent-v0e",
            "vocab_size": int(self.vocab_size),
            "embedding_dim": int(self.embedding_dim),
            "max_horizons": int(self.max_horizons),
            "encoder_layers": int(self.encoder_layers),
            "heads": int(self.heads),
            "max_len": int(self.max_len),
            "ff_mult": int(self.ff_mult),
            "dropout": float(self.dropout),
            "use_layer_norm": bool(self.use_layer_norm),
            "predictor_hidden_mult": int(self.predictor_hidden_mult),
            "target_encoder_mode": self.target_encoder_mode,
            "autoregression_mode": "direct_multi_horizon_transformer_heads",
            "horizon_conditioning": "horizon_specific_mlp_heads",
            "target_encoder": self.target_encoder_mode,
        }


def make_padding_mask(ids: Any) -> Any:
    return ids != 0


class TransformerHorizonAutoregressor:  # pragma: no cover - concrete class created in __new__
    """Small direct multi-horizon latent autoregressor.

    The model is intentionally narrow for BP-CLINJEPA-006: a shared token
    sequence encoder embeds context, and horizon-specific prediction heads map
    a context latent directly to each future horizon. Target latents can use the
    legacy shared encoder for backward compatibility or a fixed mean-token target
    space that prevents the target geometry itself from collapsing. This tests
    latent autoregression/readout only; it is not a renderer, decoder,
    generated-sequence model, or clinical/treatment model.
    """

    def __new__(cls, config: TransformerAutoregConfig | None = None, **kwargs: Any):
        torch, nn = _require_torch()
        import torch.nn.functional as F

        if config is None:
            config = TransformerAutoregConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either config or keyword arguments, not both")

        class _SequenceEncoder(nn.Module):
            def __init__(self, cfg: TransformerAutoregConfig) -> None:
                super().__init__()
                self.cfg = cfg
                self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)
                self.position_embedding = nn.Embedding(cfg.max_len, cfg.embedding_dim)
                if cfg.encoder_layers > 0:
                    layer = nn.TransformerEncoderLayer(
                        d_model=cfg.embedding_dim,
                        nhead=cfg.heads,
                        dim_feedforward=cfg.embedding_dim * cfg.ff_mult,
                        dropout=cfg.dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                    self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.encoder_layers)
                else:
                    self.transformer = None
                self.norm = nn.LayerNorm(cfg.embedding_dim) if cfg.use_layer_norm else nn.Identity()

            def forward(self, ids: Any) -> Any:
                mask = make_padding_mask(ids)
                n = ids.shape[1]
                pos = torch.arange(n, device=ids.device).clamp_max(self.cfg.max_len - 1)
                x = self.token_embedding(ids) + self.position_embedding(pos)[None, :, :]
                x = x * mask.float().unsqueeze(-1)
                if self.transformer is not None:
                    x = self.transformer(x, src_key_padding_mask=~mask)
                pooled = (x * mask.float().unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                return self.norm(pooled)

        class _FixedMeanTokenTargetEncoder(nn.Module):
            def __init__(self, cfg: TransformerAutoregConfig) -> None:
                super().__init__()
                self.cfg = cfg
                self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(20260526)
                with torch.no_grad():
                    weight = torch.randn((cfg.vocab_size, cfg.embedding_dim), generator=generator) / (cfg.embedding_dim ** 0.5)
                    weight[0].zero_()
                    weight[1:] = F.normalize(weight[1:], dim=1)
                    self.token_embedding.weight.copy_(weight)
                self.token_embedding.weight.requires_grad_(False)

            def forward(self, ids: Any) -> Any:
                mask = make_padding_mask(ids)
                x = self.token_embedding(ids) * mask.float().unsqueeze(-1)
                return x.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)

        def _effective_rank_torch(x: Any) -> Any:
            if x.shape[0] < 2:
                return torch.tensor(0.0, device=x.device)
            centered = x - x.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(centered)
            total = s.sum()
            if float(total.detach().cpu()) <= 1e-12:
                return torch.tensor(0.0, device=x.device)
            p = s / total.clamp_min(1e-12)
            return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())

        def _offdiag_cosine_mean(x: Any) -> Any:
            if x.shape[0] < 2:
                return torch.tensor(1.0, device=x.device)
            z = F.normalize(x, dim=1)
            sim = z @ z.T
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            return sim[mask].mean()

        def _latent_diagnostics_torch(x: Any) -> dict[str, float]:
            flat = x.reshape(-1, x.shape[-1])
            variance = flat.var(dim=0).mean()
            return {
                "variance_mean": float(variance.detach().cpu()),
                "effective_rank": float(_effective_rank_torch(flat).detach().cpu()),
                "offdiag_cosine_mean": float(_offdiag_cosine_mean(flat).detach().cpu()),
            }

        class _TransformerHorizonAutoregressor(nn.Module):
            def __init__(self, cfg: TransformerAutoregConfig) -> None:
                super().__init__()
                self.config = cfg
                self.encoder = _SequenceEncoder(cfg)
                if cfg.target_encoder_mode == "fixed_mean_token":
                    self.target_encoder = _FixedMeanTokenTargetEncoder(cfg)
                else:
                    self.target_encoder = self.encoder
                hidden = cfg.embedding_dim * cfg.predictor_hidden_mult
                if cfg.predictor_hidden_mult <= 0:
                    self.horizon_heads = nn.ModuleList([nn.Linear(cfg.embedding_dim, cfg.embedding_dim) for _ in range(cfg.max_horizons)])
                else:
                    self.horizon_heads = nn.ModuleList([
                        nn.Sequential(
                            nn.LayerNorm(cfg.embedding_dim),
                            nn.Linear(cfg.embedding_dim, hidden),
                            nn.GELU(),
                            nn.Linear(hidden, cfg.embedding_dim),
                        )
                        for _ in range(cfg.max_horizons)
                    ])

            @property
            def embedding_dim(self) -> int:
                return int(self.config.embedding_dim)

            @property
            def max_horizons(self) -> int:
                return int(self.config.max_horizons)

            def encode_sequence(self, ids: Any) -> Any:
                return self.encoder(ids)

            def encode_target_sequence(self, ids: Any) -> Any:
                return self.target_encoder(ids)

            def predict_rollout_from_latent(self, context_z: Any, horizon_count: int) -> Any:
                horizon_count = int(horizon_count)
                if horizon_count <= 0:
                    raise ValueError("horizon_count must be positive")
                if horizon_count > self.max_horizons:
                    raise ValueError(f"checkpoint supports max_horizons={self.max_horizons}, requested horizon_count={horizon_count}")
                return torch.stack([self.horizon_heads[h](context_z) for h in range(horizon_count)], dim=1)

            def predict_rollout_from_context_ids(self, context_ids: Any, horizon_count: int) -> Any:
                return self.predict_rollout_from_latent(self.encode_sequence(context_ids), horizon_count)

            def encode_target_rollout_from_ids(self, target_ids_by_horizon: list[Any]) -> Any:
                return torch.stack([self.encode_target_sequence(ids) for ids in target_ids_by_horizon], dim=1)

            def latent_diagnostics(self, pred: Any, target: Any) -> dict[str, dict[str, float] | list[str]]:
                pred_diag = _latent_diagnostics_torch(pred)
                target_diag = _latent_diagnostics_torch(target)
                warnings: list[str] = []
                for prefix, diag in (("pred", pred_diag), ("target", target_diag)):
                    if diag["variance_mean"] < 1e-4:
                        warnings.append(f"{prefix}_variance_below_1e-4")
                    if diag["effective_rank"] < 2.0:
                        warnings.append(f"{prefix}_effective_rank_below_2")
                    if diag["offdiag_cosine_mean"] > 0.95:
                        warnings.append(f"{prefix}_offdiag_cosine_above_0.95")
                return {"pred": pred_diag, "target": target_diag, "warnings": warnings}

            def training_loss(
                self,
                context_ids: Any,
                target_ids_by_horizon: list[Any],
                *,
                variance_floor: float = 0.05,
                variance_weight: float = 0.01,
                target_variance_weight: float = 0.0,
                wrong_horizon_weight: float = 0.0,
            ) -> tuple[Any, dict[str, float | list[str]]]:
                pred = self.predict_rollout_from_context_ids(context_ids, len(target_ids_by_horizon))
                with torch.no_grad():
                    target = self.encode_target_rollout_from_ids(target_ids_by_horizon)
                dim = pred.shape[-1]
                cosine = F.cosine_similarity(pred.reshape(-1, dim), target.reshape(-1, dim), dim=-1)
                cosine_loss = 1.0 - cosine.mean()
                pred_variance_tensor = pred.reshape(-1, dim).var(dim=0).mean()
                target_variance_tensor = target.reshape(-1, dim).var(dim=0).mean()
                pred_variance_penalty = torch.relu(torch.tensor(float(variance_floor), device=pred.device) - pred_variance_tensor)
                target_variance_penalty = torch.relu(torch.tensor(float(variance_floor), device=pred.device) - target_variance_tensor)
                loss = cosine_loss + float(variance_weight) * pred_variance_penalty + float(target_variance_weight) * target_variance_penalty
                pred_diag = _latent_diagnostics_torch(pred.detach())
                target_diag = _latent_diagnostics_torch(target.detach())
                wrong_horizon_margin = torch.tensor(0.0, device=pred.device)
                if wrong_horizon_weight and pred.shape[1] > 1:
                    aligned = cosine.reshape(pred.shape[0], pred.shape[1])
                    wrong_terms = []
                    for h in range(pred.shape[1] - 1):
                        wrong = F.cosine_similarity(pred[:, h, :], target[:, h + 1, :], dim=-1)
                        wrong_terms.append(torch.relu(0.05 + wrong - aligned[:, h]).mean())
                    wrong_horizon_margin = torch.stack(wrong_terms).mean()
                    loss = loss + float(wrong_horizon_weight) * wrong_horizon_margin
                collapse = self.latent_diagnostics(pred.detach(), target.detach())
                metrics: dict[str, float | list[str]] = {
                    "loss": float(loss.detach().cpu()),
                    "cosine_mean": float(cosine.mean().detach().cpu()),
                    "pred_variance_mean": float(pred_diag["variance_mean"]),
                    "target_variance_mean": float(target_diag["variance_mean"]),
                    "pred_effective_rank": float(pred_diag["effective_rank"]),
                    "target_effective_rank": float(target_diag["effective_rank"]),
                    "pred_offdiag_cosine_mean": float(pred_diag["offdiag_cosine_mean"]),
                    "target_offdiag_cosine_mean": float(target_diag["offdiag_cosine_mean"]),
                    "wrong_horizon_margin": float(wrong_horizon_margin.detach().cpu()),
                    "collapse_warnings": collapse["warnings"],
                }
                return loss, metrics

            def architecture_metadata(self) -> dict[str, Any]:
                return self.config.to_dict()

        return _TransformerHorizonAutoregressor(config)


def build_transformer_autoreg_from_checkpoint(checkpoint: dict[str, Any]) -> Any:
    config = TransformerAutoregConfig.from_checkpoint(checkpoint)
    model = TransformerHorizonAutoregressor(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def checkpoint_metadata(config: TransformerAutoregConfig, *, horizon_count_trained: int) -> dict[str, Any]:
    return {
        "model_family": "transformer_autoregressive_latent_v0e",
        "architecture": config.to_dict(),
        "vocab_size": int(config.vocab_size),
        "embedding_dim": int(config.embedding_dim),
        "max_horizons": int(config.max_horizons),
        "horizon_count_trained": int(horizon_count_trained),
        "target_encoder_mode": config.target_encoder_mode,
        "autoregression_mode": "direct_multi_horizon_transformer_heads",
        "aggregate_only": True,
    }
