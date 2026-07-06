from __future__ import annotations

from typing import Any

SUPPORTED_AUTOREGRESSION_MODES = ("recursive", "horizon_conditioned")
DEFAULT_AUTOREGRESSION_MODE = "recursive"


def normalize_autoregression_mode(mode: str | None) -> str:
    value = (mode or DEFAULT_AUTOREGRESSION_MODE).strip().lower().replace("-", "_")
    aliases = {
        "recursive_v0b": "recursive",
        "mean_token_recursive": "recursive",
        "horizon_heads": "horizon_conditioned",
        "horizon_specific_heads": "horizon_conditioned",
        "horizon_conditioned_heads": "horizon_conditioned",
        "time_conditioned": "horizon_conditioned",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_AUTOREGRESSION_MODES:
        raise ValueError(f"Unsupported autoregression mode {mode!r}; expected one of {SUPPORTED_AUTOREGRESSION_MODES}")
    return value


def checkpoint_autoregression_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return safe architecture metadata for a mean-token v0B checkpoint."""
    architecture = checkpoint.get("architecture") or {}
    mode = normalize_autoregression_mode(
        checkpoint.get("autoregression_mode")
        or architecture.get("autoregression_mode")
        or architecture.get("prediction_mode")
    )
    horizon_count_trained = int(
        checkpoint.get("horizon_count_trained")
        or architecture.get("horizon_count_trained")
        or architecture.get("horizon_count")
        or 1
    )
    max_horizons = int(checkpoint.get("max_horizons") or architecture.get("max_horizons") or horizon_count_trained or 1)
    return {
        "autoregression_mode": mode,
        "horizon_count_trained": horizon_count_trained,
        "max_horizons": max_horizons,
        "horizon_conditioning": "horizon_specific_linear_heads" if mode == "horizon_conditioned" else "none_recursive_predictor_reuse",
    }


class MeanTokenJEPA:  # pragma: no cover - real class is created after torch import in __new__
    """Type placeholder so importers get a helpful error if torch is absent."""

    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - depends on optional torch
            raise ImportError("MeanTokenJEPA requires the optional torch dependency") from exc

        class _MeanTokenJEPA(nn.Module):
            """Mean-token v0B model with explicit horizon-conditioned rollout support.

            `recursive` is backward-compatible with the original v0B checkpoint
            structure: an embedding table plus a single `predictor` MLP reused for
            one-step prediction and recursive rollout.

            `horizon_conditioned` keeps the same embedding table but replaces
            recursive reuse with explicit horizon-specific linear heads from the
            context latent. This is a deliberately minimal BP-CLINJEPA-003 repair
            path: it can represent horizon/time-specific futures and should fail
            loudly if asked to export more horizons than the checkpoint declares.
            """

            DEFAULT_EMPTY_PROTOTYPE_SEED = 20260706

            def __init__(
                self,
                vocab_size: int,
                embedding_dim: int,
                *,
                autoregression_mode: str = DEFAULT_AUTOREGRESSION_MODE,
                max_horizons: int = 1,
                encode_empty: bool = True,
                empty_prototype_seed: int | None = None,
            ) -> None:
                super().__init__()
                self.vocab_size = int(vocab_size)
                self.embedding_dim = int(embedding_dim)
                self.autoregression_mode = normalize_autoregression_mode(autoregression_mode)
                self.max_horizons = max(1, int(max_horizons))
                self.encode_empty = bool(encode_empty)
                self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
                self.predictor = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                )
                if self.autoregression_mode == "horizon_conditioned":
                    self.horizon_heads = nn.ModuleList(
                        [nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.max_horizons)]
                    )
                # Encode-empty (Pi R4): a FROZEN, seeded, unit-norm silence prototype
                # (register_buffer, not a Parameter => immovable under no_grad, excluded
                # from the optimizer), plus supervised occupancy/count "hurdle" heads
                # that are the AUTHORITATIVE empty/non-empty + count-0 owner. z_empty is
                # auxiliary latent geometry only.
                self.empty_prototype_seed = int(
                    empty_prototype_seed if empty_prototype_seed is not None else self.DEFAULT_EMPTY_PROTOTYPE_SEED
                )
                if self.encode_empty:
                    self.register_buffer("empty_prototype", self._draw_prototype(self.empty_prototype_seed))

                    def _scalar_head() -> Any:
                        return nn.Sequential(
                            nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(),
                            nn.Linear(self.embedding_dim, 1),
                        )

                    if self.autoregression_mode == "horizon_conditioned":
                        self.occupancy_heads = nn.ModuleList([_scalar_head() for _ in range(self.max_horizons)])
                        self.count_heads = nn.ModuleList([_scalar_head() for _ in range(self.max_horizons)])
                    else:
                        self.occupancy_head = _scalar_head()
                        self.count_head = _scalar_head()

            def _draw_prototype(self, seed: int) -> Any:
                """Deterministic unit-norm random vector (frozen silence anchor)."""
                g = torch.Generator().manual_seed(int(seed))
                with torch.no_grad():
                    v = torch.randn(self.embedding_dim, generator=g)
                    return v / v.norm().clamp_min(1e-12)

            def reseed_empty_prototype(self, seed: int) -> None:
                """Re-draw the frozen prototype (used by the init separation check in
                training: reseed if it lands too close to the non-empty target sample)."""
                if not self.encode_empty:
                    return
                self.empty_prototype_seed = int(seed)
                with torch.no_grad():
                    self.empty_prototype.copy_(self._draw_prototype(seed).to(self.empty_prototype.device))

            def mean_embed(self, ids: Any) -> Any:
                mask = (ids != 0).float().unsqueeze(-1)
                emb = self.embedding(ids) * mask
                return emb.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

            def target_latent(self, ids: Any, is_empty: Any = None) -> Any:
                """Target latent z+: mean_embed(ids) for populated rows, the frozen
                empty_prototype for empty (silence) rows. Bypasses the all-[PAD]
                mean_embed zero-vector / NaN-cosine collision for empties."""
                z = self.mean_embed(ids)
                if is_empty is None or not self.encode_empty:
                    return z
                cond = is_empty.to(dtype=torch.bool, device=z.device).view(-1, 1)
                return torch.where(cond, self.empty_prototype.to(z.dtype).view(1, -1), z)

            def predict_occupancy_from_latent(self, context_z: Any, horizon_count: int) -> Any:
                """Return (occupancy_logit[B,H,1], count_pred[B,H,1]) off the rolled-out
                latent — the authoritative supervised empty/non-empty + count signal."""
                if not self.encode_empty:
                    raise RuntimeError("occupancy head requires encode_empty=True")
                rollout = self.predict_rollout_from_latent(context_z, horizon_count)  # [B,H,D]
                if self.autoregression_mode == "horizon_conditioned":
                    occ = torch.stack([self.occupancy_heads[h](rollout[:, h, :]) for h in range(horizon_count)], dim=1)
                    cnt = torch.stack([self.count_heads[h](rollout[:, h, :]) for h in range(horizon_count)], dim=1)
                    return occ, cnt
                return self.occupancy_head(rollout), self.count_head(rollout)

            def predict_rollout_from_latent(self, context_z: Any, horizon_count: int) -> Any:
                horizon_count = int(horizon_count)
                if horizon_count <= 0:
                    raise ValueError("horizon_count must be positive")
                if self.autoregression_mode == "recursive":
                    pred = self.predictor(context_z)
                    steps = [pred]
                    for _ in range(1, horizon_count):
                        pred = self.predictor(pred)
                        steps.append(pred)
                    return torch.stack(steps, dim=1)
                if horizon_count > self.max_horizons:
                    raise ValueError(
                        f"horizon-conditioned checkpoint supports max_horizons={self.max_horizons}, "
                        f"but horizon_count={horizon_count} was requested"
                    )
                return torch.stack([self.horizon_heads[h](context_z) for h in range(horizon_count)], dim=1)

            def predict_rollout_from_context_ids(self, ids: Any, horizon_count: int) -> Any:
                return self.predict_rollout_from_latent(self.mean_embed(ids), horizon_count)

            def predict_from_context_ids(self, ids: Any) -> Any:
                """Backward-compatible one-step prediction used by older callers."""
                return self.predict_rollout_from_context_ids(ids, 1)[:, 0, :]

            def architecture_metadata(self) -> dict[str, Any]:
                meta = {
                    "name": "mean-token-jepa-v0",
                    "embedding_dim": self.embedding_dim,
                    "predictor": "2-layer-mlp" if self.autoregression_mode == "recursive" else "horizon-specific-linear-heads",
                    "target_encoder": "shared_embedding_stop_gradient"
                    + ("+frozen_empty_prototype" if self.encode_empty else ""),
                    "autoregression_mode": self.autoregression_mode,
                    "max_horizons": self.max_horizons,
                    "horizon_conditioning": "horizon_specific_linear_heads"
                    if self.autoregression_mode == "horizon_conditioned"
                    else "none_recursive_predictor_reuse",
                    "encode_empty": self.encode_empty,
                }
                if self.encode_empty:
                    meta["empty_prototype_seed"] = self.empty_prototype_seed
                    meta["occupancy_head"] = "hurdle_bce_plus_log1p_count"
                return meta

        return _MeanTokenJEPA(*args, **kwargs)


def build_mean_token_jepa_from_checkpoint(checkpoint: dict[str, Any]) -> Any:
    """Instantiate a mean-token v0B model from checkpoint metadata.

    Old checkpoints without BP003 metadata load as `recursive`. New checkpoints
    that declare `horizon_conditioned` must include the horizon-head weights;
    otherwise `load_state_dict(..., strict=True)` fails clearly.
    """
    config = checkpoint_autoregression_config(checkpoint)
    sd = checkpoint["model_state_dict"]
    # Encode-empty is inferred from the checkpoint contents so old checkpoints
    # (no empty_prototype / occupancy heads) still load strict=True.
    encode_empty = any(str(k) == "empty_prototype" or str(k).startswith(("occupancy_head", "count_head")) for k in sd)
    arch = checkpoint.get("architecture") or {}
    model = MeanTokenJEPA(
        int(checkpoint["vocab_size"]),
        int(checkpoint["embedding_dim"]),
        autoregression_mode=config["autoregression_mode"],
        max_horizons=config["max_horizons"],
        encode_empty=encode_empty,
        empty_prototype_seed=arch.get("empty_prototype_seed"),
    )
    model.load_state_dict(sd, strict=True)
    return model
