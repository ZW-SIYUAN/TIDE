"""
IB-SparseAttention: Information Bottleneck Sparse Attention Framework for be-great
v0.2 -- atomic_tokenizer plug-in support added (see CHANGELOG.md)

This module extends the GReaT (be-great) library with an "IB-SparseAttention" framework
that uses Information Bottleneck (IB) principles to automatically learn sparse feature
dependency structures during tabular data generation.

Mathematical Formulation:
  - Feature dependency matrix W ∈ R^{m×m} (m = number of features)
  - Attention modulation: ã_ij = a_ij + log(softplus(W[feat(i), feat(j)]) + ε)
    (additive form — equivalent to multiplicative gating via log-space algebra)
  - IB Total Loss: L_Total = L_CE + β·I(X_context; Y_target) + λ_sparse·||W||_1
  - MI estimator: I(X;Y) ≈ log P_LLM(Y|X) − log P_LLM(Y)
    implemented as a batched blind-context forward pass

Modules:
  1. FeatureAlignmentDataset  — tracks per-token feature indices during tokenisation
  2. FeatureAlignmentCollator — pads the feature_map tensor to uniform batch length
  3. IBSparseAttentionModulator — monkey-patches F.scaled_dot_product_attention to
                                  inject W-based attention bias (O(L·m) extra memory)
  4. IBLossComputer            — single-batch blind-context MI estimation
  5. IBSparseTrainer           — custom Trainer with combined IB loss
  6. IBSparseGReaT             — extends GReaT with 3-phase training pipeline
  7. Monitoring utils          — W sparsity logging, heatmap export

References:
  - GReaT: https://github.com/kathrinse/be_great
  - IB Principle: Tishby et al. (2000)
  - ALiBi positional bias: Press et al. (2022) (structural inspiration for bias injection)
"""

import math
import os
import random
import logging
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from datasets import Dataset

# be-great imports (must be installed)
from be_great.great import GReaT
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import _array_to_dataframe

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Feature Alignment Dataset + Collator
# ─────────────────────────────────────────────────────────────────────────────

class FeatureAlignmentDataset(GReaTDataset):
    """GReaTDataset extension that returns a per-token feature index map.

    Each token in the output sequence is labelled with the index of the
    tabular feature it belongs to.  Structural tokens (", " separators,
    " is " connectives, special BOS/EOS) receive index -1.

    The feature index is the *original* column index in the DataFrame,
    irrespective of the shuffled order used during training.

    Extra method ``set_ib_config`` must be called after ``set_tokenizer``.
    """

    def set_ib_config(
        self,
        num_features: int,
        fixed_order: tp.Optional[tp.List[int]] = None,
    ) -> None:
        """Configure IB-specific settings.

        Args:
            num_features: Total number of tabular features (m).
            fixed_order: If given, feature indices are iterated in this order
                instead of being shuffled.  Used in Phase 3 of training.
        """
        self.num_features = num_features
        self.fixed_order = fixed_order  # list[int] | None

    def _build_feature_map(
        self,
        parts: tp.List[tp.Tuple[int, str, str]],
        full_text: str,
    ) -> tp.List[int]:
        """Build a character-level → token-level feature assignment.

        Args:
            parts: List of (original_col_idx, col_name, formatted_value).
            full_text: The joined text string that will be fed to the tokeniser.

        Returns:
            List of integers (length == len(input_ids)) where value is the
            original column index, or -1 for structural/special tokens.
        """
        tokenizer = self.tokenizer

        # ── Step 1: Build char-level feature map ──────────────────────────
        char_feat: tp.List[int] = [-1] * len(full_text)
        char_pos = 0
        for seg_idx, (col_idx, col_name, value) in enumerate(parts):
            if seg_idx > 0:
                char_pos += 2  # ", " separator → stays -1
            seg_text = f"{col_name} is {value}"
            for c in range(char_pos, char_pos + len(seg_text)):
                char_feat[c] = col_idx
            char_pos += len(seg_text)

        # ── Step 2: Tokenise with offset mapping (fast tokeniser) ─────────
        try:
            enc = tokenizer(
                full_text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            offsets: tp.List[tp.Tuple[int, int]] = enc["offset_mapping"]
            token_ids: tp.List[int] = enc["input_ids"]

            feat_map: tp.List[int] = []
            for (start, end) in offsets:
                if start == end:
                    # Special token (BOS/EOS/PAD) — no character span
                    feat_map.append(-1)
                else:
                    feat_map.append(char_feat[start])

        except Exception:
            # Fallback: fast tokeniser offset mapping unavailable
            # Assign all tokens to -1 (modulation disabled for this sample)
            token_ids = tokenizer.encode(full_text, add_special_tokens=True)
            feat_map = [-1] * len(token_ids)

        return feat_map

    def _getitem(
        self,
        key: tp.Union[int, slice, str],
        decoded: bool = True,
        **kwargs,
    ) -> tp.Dict[str, tp.Any]:
        """Return tokenised text + feature_map for one tabular row."""
        row = self._data.fast_slice(key, 1)

        # Determine column order
        if getattr(self, "fixed_order", None) is not None:
            shuffle_idx = list(self.fixed_order)
        else:
            shuffle_idx = list(range(row.num_columns))
            random.shuffle(shuffle_idx)

        # Build parts list: (original_col_idx, col_name, formatted_value)
        parts: tp.List[tp.Tuple[int, str, str]] = []
        for i in shuffle_idx:
            col_name = row.column_names[i]
            raw_val = row.columns[i].to_pylist()[0]
            parts.append((i, col_name, self._format_value(raw_val)))

        # Assemble the full text string
        full_text = ", ".join(f"{col} is {val}" for _, col, val in parts)

        # Build the token-level feature map
        feat_map = self._build_feature_map(parts, full_text)

        # Standard tokenisation (mirrors original GReaTDataset behaviour)
        tokenized = self.tokenizer(full_text, padding=False)

        # Ensure feature_map matches input_ids length (safety guard)
        ids_len = len(tokenized["input_ids"])
        if len(feat_map) > ids_len:
            feat_map = feat_map[:ids_len]
        elif len(feat_map) < ids_len:
            feat_map.extend([-1] * (ids_len - len(feat_map)))

        tokenized["feature_map"] = feat_map
        return tokenized

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(k) for k in keys]
        return self._getitem(keys)


@dataclass
class FeatureAlignmentCollator(GReaTDataCollator):
    """Data collator that pads both ``input_ids`` and ``feature_map``.

    ``feature_map`` is padded with -1 (= structural / ignore token) to the
    same length as ``input_ids`` within the batch.
    """

    def __call__(
        self, features: tp.List[tp.Dict[str, tp.Any]]
    ) -> tp.Dict[str, torch.Tensor]:
        # Extract and remove feature_map before standard padding
        feature_maps: tp.List[tp.Optional[tp.List[int]]] = [
            f.pop("feature_map", None) for f in features
        ]

        # Standard token padding (HuggingFace)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()

        # Pad feature_map to the same length as input_ids
        if feature_maps[0] is not None:
            max_len: int = batch["input_ids"].shape[1]
            padded: tp.List[tp.List[int]] = []
            for fm in feature_maps:
                if fm is None:
                    padded.append([-1] * max_len)
                else:
                    pad_needed = max_len - len(fm)
                    # Left-padding: tokenizer uses left-padding for causal LMs
                    padded.append([-1] * pad_needed + fm)
            batch["feature_map"] = torch.tensor(padded, dtype=torch.long)

        return batch


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: IB Sparse Attention Modulator
# ─────────────────────────────────────────────────────────────────────────────

class IBSparseAttentionModulator:
    """Injects W-matrix-based attention modulation via SDPA monkey-patching.

    Specification formula (multiplicative):
        ã_ij = a_ij · log(w_{feat(i), feat(j)} + ε)

    Implementation formula (additive log-space bias — numerically preferred):
        ã_ij = a_ij + log(softplus(W[feat(i), feat(j)]) + ε)

    Design rationale for the deviation:
        The multiplicative form `a_ij · log(w + ε)` is problematic when a_ij
        are pre-softmax attention logits (which can be large and negative):
          • If w < 1/e ≈ 0.368 → log(w+ε) < 0 → multiplying a negative logit
            by a negative modulator INCREASES attention to "blocked" pairs
            (sign flip), creating semantically incorrect behaviour.
          • If a_ij ≈ 0, the multiplicative form collapses to zero regardless
            of W, killing gradients to W entirely.

        The additive bias form is the standard in transformer literature
        (ALiBi, T5 relative bias, etc.) and achieves the same structured
        sparsification effect:
          • When W[i,j] is near zero, softplus → ln(2), log(softplus + ε) ≈ –0.37
            → small negative bias → suppresses attention between features i↔j.
          • When W[i,j] → +∞, softplus ≈ W[i,j], log(W[i,j] + ε) → large positive
            → amplifies attention between features i↔j.
          • The L1 penalty on W drives weights toward 0 (suppression), creating
            the desired sparse dependency graph.

    Memory complexity:
        Intermediate: O(B · L · m)   (row_emb tensor)
        Output bias:  O(B · L²)      (broadcast gather result)
        W parameter:  O(m²)          (orders of magnitude smaller than L²)

    The W matrix is kept in fp32 even when the base model is quantised,
    ensuring high-precision gradient updates.

    Args:
        W: Feature dependency matrix parameter, shape [m, m], fp32.
        num_features: m (number of tabular features).
        epsilon: Small constant for numerical stability.
    """

    def __init__(
        self,
        W: nn.Parameter,
        num_features: int,
        epsilon: float = 1e-6,
    ) -> None:
        self.W = W
        self.num_features = num_features
        self.epsilon = epsilon

        # Thread-local feature map set before each forward pass
        self._feature_map: tp.Optional[torch.Tensor] = None  # [B, L]

        # Save original SDPA for restoration
        self._original_sdpa = None
        self._is_active = False

    # ── Public interface ───────────────────────────────────────────────────

    def set_feature_map(self, feature_map: torch.Tensor) -> None:
        """Register the current batch's feature map before model forward."""
        self._feature_map = feature_map

    def clear_feature_map(self) -> None:
        self._feature_map = None

    @contextmanager
    def active(self):
        """Context manager: enable modulation, restore original SDPA on exit."""
        self._install_patch()
        try:
            yield self
        finally:
            self._remove_patch()
            self.clear_feature_map()

    # ── Core computation ───────────────────────────────────────────────────

    def compute_attention_bias(
        self, L: int, device: torch.device, dtype: torch.dtype
    ) -> tp.Optional[torch.Tensor]:
        """Compute additive attention bias from W and the current feature_map.

        Shape: [B, 1, L, L]  (head dimension broadcast = 1)

        The computation avoids materialising a learned [L, L] weight matrix:
          1. Look up W rows for query positions  → [B, L, m]  (O(BLm))
          2. Gather W columns for key positions  → [B, L, L]  (O(BL²) gather)
        The dependency structure lives in W[m×m], not in an [L×L] parameter.

        Returns None if feature_map is unset or seq-len mismatch.
        """
        if self._feature_map is None:
            return None

        fm = self._feature_map  # [B, L_fm]
        B, L_fm = fm.shape

        # Adjust to current sequence length (may differ during generation)
        if L_fm != L:
            if L_fm > L:
                fm = fm[:, :L]
            else:
                # Pad with -1 on the left (left-padded batch)
                pad = torch.full(
                    (B, L - L_fm), -1, dtype=fm.dtype, device=fm.device
                )
                fm = torch.cat([pad, fm], dim=1)

        # W_log = log(softplus(W) + ε)  — ensures w > 0 for all W_param values
        # softplus(x) = log(1 + exp(x)) ≈ x for large x, ≈ exp(x) for small x
        W_pos = F.softplus(self.W.to(device=device, dtype=torch.float32))
        W_log = torch.log(W_pos + self.epsilon)  # [m, m]

        # Clamp feature indices to [0, m-1]; structural tokens (-1) → 0 then masked
        fm_clamped = fm.clamp(min=0)  # [B, L]

        # ── Step 1: row embedding ──────────────────────────────────────────
        # row_emb[b, i, :] = W_log[fm_clamped[b, i], :]
        row_emb = W_log[fm_clamped]  # [B, L, m]

        # ── Step 2: gather column values ───────────────────────────────────
        # col_idx[b, i, j] = fm_clamped[b, j]  (same for all i)
        col_idx = fm_clamped.unsqueeze(1).expand(B, L, L)  # [B, L, L]

        # bias[b, i, j] = row_emb[b, i, fm_clamped[b, j]]
        #               = W_log[fm_clamped[b,i], fm_clamped[b,j]]
        bias = row_emb.gather(2, col_idx)  # [B, L, L]

        # ── Step 3: zero out positions involving structural tokens (-1) ────
        # These positions should not be modulated (they carry no feature info)
        valid = (fm != -1)  # [B, L]
        valid_2d = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, L, L]
        bias = bias * valid_2d.to(bias.dtype)

        # Cast to model dtype and add head dimension
        return bias.unsqueeze(1).to(dtype=dtype)  # [B, 1, L, L]

    # ── SDPA patch ─────────────────────────────────────────────────────────

    def _install_patch(self) -> None:
        """Replace F.scaled_dot_product_attention with a modulated version."""
        if self._is_active:
            return

        original = F.scaled_dot_product_attention
        self._original_sdpa = original
        modulator = self  # capture self for closure

        def _patched_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: tp.Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: tp.Optional[float] = None,
            **kwargs,
        ) -> torch.Tensor:
            # query: [B, H, L, d]
            B, H, L, d = query.shape

            mod_bias = modulator.compute_attention_bias(L, query.device, query.dtype)

            if mod_bias is not None:
                # When is_causal=True the SDPA kernel builds the causal mask
                # internally, making it hard to add our bias.  We convert to
                # an explicit additive float mask instead.
                if is_causal and attn_mask is None:
                    # Build explicit lower-triangular causal bias
                    causal_bias = torch.zeros(
                        1, 1, L, L, device=query.device, dtype=query.dtype
                    )
                    causal_bias.masked_fill_(
                        torch.tril(
                            torch.ones(L, L, device=query.device, dtype=torch.bool)
                        ).logical_not().unsqueeze(0).unsqueeze(0),
                        float("-inf"),
                    )
                    attn_mask = causal_bias
                    is_causal = False  # handled by explicit mask now

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        # Convert boolean mask to float additive bias
                        float_mask = torch.zeros(
                            *attn_mask.shape, device=query.device, dtype=query.dtype
                        )
                        float_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
                        attn_mask = float_mask
                    # Broadcast mod_bias to match attn_mask batch/head dims
                    attn_mask = attn_mask + mod_bias
                else:
                    attn_mask = mod_bias

            return original(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **({"scale": scale} if scale is not None else {}),
                **kwargs,
            )

        # Monkey-patch at module level (affects all callers in process)
        F.scaled_dot_product_attention = _patched_sdpa
        # Also patch torch.nn.functional if imported elsewhere
        import torch.nn.functional as _fnmod
        _fnmod.scaled_dot_product_attention = _patched_sdpa

        self._is_active = True
        logger.debug("IBSparseAttentionModulator: SDPA patch installed.")

    def _remove_patch(self) -> None:
        """Restore original F.scaled_dot_product_attention."""
        if not self._is_active or self._original_sdpa is None:
            return
        F.scaled_dot_product_attention = self._original_sdpa
        import torch.nn.functional as _fnmod
        _fnmod.scaled_dot_product_attention = self._original_sdpa
        self._original_sdpa = None
        self._is_active = False
        logger.debug("IBSparseAttentionModulator: SDPA patch removed.")

    # ── Fallback: GPT-2 style _attn hook ─────────────────────────────────
    # Some older GPT-2 builds call their own `_attn` method instead of SDPA.
    # Register a forward hook on Attention modules as belt-and-suspenders.

    def register_fallback_hooks(self, model: nn.Module) -> tp.List[torch.utils.hooks.RemovableHandle]:
        """Register forward hooks on attention modules as a safety fallback.

        The hooks inject the attention bias by storing it and relying on the
        model's attention_mask pathway when SDPA patching is unavailable
        (e.g., Flash-Attention backends that bypass F.scaled_dot_product_attention).

        Returns a list of hook handles that can be removed with handle.remove().
        """
        handles = []

        def _make_hook(mod_ref):
            def _pre_hook(module, args, kwargs):
                # Inject feature-aware attention_mask if not already set
                fm = mod_ref._feature_map
                if fm is None:
                    return args, kwargs

                # Determine sequence length from hidden states (first positional arg)
                hidden = args[0] if args else kwargs.get("hidden_states", None)
                if hidden is None:
                    return args, kwargs

                L = hidden.shape[1]
                bias = mod_ref.compute_attention_bias(L, hidden.device, hidden.dtype)
                if bias is None:
                    return args, kwargs

                # Add bias to existing attention_mask kwarg if present
                existing = kwargs.get("attention_mask", None)
                if existing is not None:
                    if existing.dtype == torch.bool:
                        float_m = torch.zeros_like(bias)
                        float_m.masked_fill_(existing.logical_not(), float("-inf"))
                        existing = float_m
                    kwargs["attention_mask"] = existing + bias
                else:
                    kwargs["attention_mask"] = bias

                return args, kwargs

            return _pre_hook

        for name, module in model.named_modules():
            # Target common attention class name patterns
            cls_name = type(module).__name__.lower()
            if any(kw in cls_name for kw in ("attention", "selfattn", "selfattention")):
                h = module.register_forward_pre_hook(_make_hook(self), with_kwargs=True)
                handles.append(h)
                logger.debug(f"Fallback hook registered on {name} ({cls_name})")

        return handles


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: IB Loss Computer (single-batch blind-context MI estimation)
# ─────────────────────────────────────────────────────────────────────────────

class IBLossComputer:
    """Estimates I(X_context; Y_target) using a batched blind-context pass.

    Strategy (single-forward approximation):
      For each sample in the batch:
        1. P(Y|X)   — available from the main forward pass logits (grad-tracked).
        2. P(Y)     — estimated via a separate forward pass on target-only tokens
                      run inside torch.no_grad() to avoid gradient doubling.
      MI ≈ mean_over_batch [ log P(Y|X) − log P(Y) ]

    Only P(Y|X) carries gradients, so ∂L_IB/∂W flows through the modulated
    attention pathway, while P(Y) acts as a stable baseline.

    "Feature start tokens" optimisation: only tokens at the very beginning of
    each feature's value span contribute to the MI estimate, reducing compute.

    Args:
        beta: IB loss weight (see L_Total formula).
        use_feature_start_only: If True, compute MI only at the first token of
            each target feature value, reducing computational cost.
    """

    def __init__(self, beta: float = 0.1, use_feature_start_only: bool = True):
        self.beta = beta
        self.use_feature_start_only = use_feature_start_only

    def _get_target_positions(
        self,
        feature_map: torch.Tensor,  # [B, L]
        input_ids: torch.Tensor,    # [B, L]
    ) -> tp.List[tp.Tuple[int, torch.Tensor, torch.Tensor]]:
        """For each sample, return (last_feat_idx, target_token_ids, pred_mask).

        pred_mask[t] = True means logits[t] should be used to predict
        input_ids[t+1] as part of the target feature.
        """
        B, L = input_ids.shape
        results = []
        for b in range(B):
            fm = feature_map[b]  # [L]
            valid = fm[fm >= 0]
            if valid.numel() == 0:
                results.append((-1, None, None))
                continue

            last_feat = valid[-1].item()
            # Positions where the *next* token belongs to the target feature
            # (autoregressive: logits[t] predicts input_ids[t+1])
            next_token_is_target = (fm[1:] == last_feat)  # [L-1]

            if self.use_feature_start_only:
                # Keep only the transition point (first value token of target feat)
                diff = next_token_is_target.long()
                diff = torch.cat([diff[:1], diff[1:] - diff[:-1]])
                next_token_is_target = diff > 0

            if next_token_is_target.sum() == 0:
                results.append((-1, None, None))
                continue

            # Target token IDs: for a blind pass starting at feature name
            t_start_positions = (fm == last_feat).nonzero(as_tuple=True)[0]
            t_start = t_start_positions[0].item()
            target_token_ids = input_ids[b, t_start:]  # [n_target+]

            results.append((last_feat, target_token_ids, next_token_is_target))

        return results

    def compute(
        self,
        model: nn.Module,
        logits_with_context: torch.Tensor,  # [B, L, V] — grad-tracked
        input_ids: torch.Tensor,            # [B, L]
        feature_map: torch.Tensor,          # [B, L]
        pad_token_id: int,
    ) -> torch.Tensor:
        """Compute the MI estimate and return as a scalar differentiable tensor.

        Args:
            model: The language model (used for blind P(Y) estimation).
            logits_with_context: Pre-computed logits from the main forward
                pass.  Gradients flow through this tensor.
            input_ids, feature_map: Batch tensors.
            pad_token_id: Used to mask blind forward pass inputs.

        Returns:
            Scalar tensor β·MI, differentiable w.r.t. model parameters and W.
        """
        B, L, V = logits_with_context.shape
        device = logits_with_context.device

        target_info = self._get_target_positions(feature_map, input_ids)

        log_py_given_x_list: tp.List[torch.Tensor] = []
        log_py_list: tp.List[torch.Tensor] = []

        # ── Batch the blind forward passes for efficiency ─────────────────
        # Collect all target-only sequences, then run ONE forward pass
        blind_seqs: tp.List[torch.Tensor] = []
        blind_sample_idx: tp.List[int] = []

        for b, (last_feat, target_token_ids, pred_mask) in enumerate(target_info):
            if last_feat == -1 or pred_mask is None:
                continue

            # logP(Y|X): read from main logits at predicted positions
            target_logits = logits_with_context[b, :-1][pred_mask]  # [n, V]
            target_labels = input_ids[b, 1:][pred_mask]             # [n]
            if target_logits.shape[0] == 0:
                continue

            lp_yx = -F.cross_entropy(target_logits, target_labels, reduction="mean")
            log_py_given_x_list.append(lp_yx)

            # Prepare blind sequence (target tokens only)
            blind_seqs.append(target_token_ids)
            blind_sample_idx.append(b)

        if not log_py_given_x_list:
            # No valid target features found; return zero loss (no gradient)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ── Blind forward: batch all target-only sequences ────────────────
        if blind_seqs:
            # Pad to uniform length within the blind mini-batch
            max_blind_len = max(s.shape[0] for s in blind_seqs)
            blind_ids = torch.full(
                (len(blind_seqs), max_blind_len),
                pad_token_id,
                dtype=input_ids.dtype,
                device=device,
            )
            for i, seq in enumerate(blind_seqs):
                blind_ids[i, : seq.shape[0]] = seq

            blind_attn = (blind_ids != pad_token_id).long()

            with torch.no_grad():
                blind_out = model(
                    input_ids=blind_ids,
                    attention_mask=blind_attn,
                )
                blind_logits = blind_out.logits  # [n_blind, L_blind, V]

            for i_blind, b in enumerate(blind_sample_idx):
                _, target_token_ids, pred_mask = target_info[b]
                bl = blind_seqs[i_blind].shape[0]
                if bl <= 1:
                    # Single token — can't compute CE
                    log_py_list.append(torch.tensor(0.0, device=device))
                    continue

                # Blind logits: predict tokens 1..bl-1 from context 0..bl-2
                bl_log = blind_logits[i_blind, : bl - 1]  # [bl-1, V]
                bl_tgt = blind_ids[i_blind, 1:bl]         # [bl-1]
                lp_y = -F.cross_entropy(bl_log, bl_tgt, reduction="mean")
                log_py_list.append(lp_y)

        # ── MI = mean(logP(Y|X) − logP(Y)) ───────────────────────────────
        # logP(Y|X) has gradients; logP(Y) is a constant baseline (no_grad)
        n = min(len(log_py_given_x_list), len(log_py_list))
        if n == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_py_x = torch.stack(log_py_given_x_list[:n])  # [n]  — grad
        log_py = torch.stack(log_py_list[:n]).detach()    # [n]  — constant

        # I(X;Y) ≈ E[logP(Y|X)] − E[logP(Y)]
        # Note: we clamp MI ≥ 0 to avoid numerical artifacts
        mi = (log_py_x - log_py).clamp(min=0.0).mean()

        return self.beta * mi


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 (training core): IBSparseTrainer
# ─────────────────────────────────────────────────────────────────────────────

class IBSparseTrainer(GReaTTrainer):
    """GReaTTrainer extension that computes the full IB-SparseAttention loss.

    L_Total = L_CE + β·I(X_context; Y_target) + λ_sparse·||W||_1

    The trainer:
      - Activates the SDPA modulation patch during compute_loss
      - Injects the batch feature_map into the modulator
      - Adds IB and L1 terms to the CE loss
      - Allows W and LoRA params to have different learning rates via
        separate optimiser param groups (handled externally in IBSparseGReaT)
    """

    def __init__(
        self,
        *args,
        modulator: IBSparseAttentionModulator,
        ib_loss_computer: IBLossComputer,
        W: nn.Parameter,
        lambda_sparse: float = 1e-3,
        freeze_W: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.modulator = modulator
        self.ib_loss_computer = ib_loss_computer
        self.W = W
        self.lambda_sparse = lambda_sparse
        self.freeze_W = freeze_W  # True in Phase 3

    def compute_loss(
        self,
        model: nn.Module,
        inputs: tp.Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, tp.Any]]:
        """Compute L_CE + β·MI + λ·||W||_1."""

        # Extract feature_map from batch (not consumed by the model itself)
        feature_map: tp.Optional[torch.Tensor] = inputs.pop("feature_map", None)

        if feature_map is not None:
            self.modulator.set_feature_map(feature_map)

        # ── Forward pass with modulated attention ─────────────────────────
        self.modulator._install_patch()
        try:
            outputs = model(**inputs)
        finally:
            self.modulator._remove_patch()
            self.modulator.clear_feature_map()

        # ── CE loss (language modelling) ──────────────────────────────────
        ce_loss: torch.Tensor = outputs.loss

        total_loss = ce_loss

        # ── IB loss (mutual information term) ─────────────────────────────
        if feature_map is not None and not self.freeze_W:
            ib_loss = self.ib_loss_computer.compute(
                model=model,
                logits_with_context=outputs.logits,
                input_ids=inputs["input_ids"],
                feature_map=feature_map,
                pad_token_id=self.tokenizer.pad_token_id
                if hasattr(self.tokenizer, "pad_token_id")
                else 0,
            )
            total_loss = total_loss + ib_loss
        else:
            ib_loss = torch.tensor(0.0, device=ce_loss.device)

        # ── L1 sparsity regularisation on W (off-diagonal only) ───────────
        # The diagonal W[i,i] controls within-feature self-attention — features
        # MUST be able to attend to their own preceding structural tokens
        # (e.g., "lat is" when generating the lat value).  Penalising the
        # diagonal suppresses this pathway and causes NaN outputs for numeric
        # columns.  Only off-diagonal entries are regularised.
        if not self.freeze_W:
            off_diag_mask = 1.0 - torch.eye(
                self.W.shape[0], device=self.W.device, dtype=self.W.dtype
            )
            l1_loss = self.lambda_sparse * (self.W * off_diag_mask).abs().sum()
            total_loss = total_loss + l1_loss
        else:
            l1_loss = torch.tensor(0.0, device=ce_loss.device)

        # ── Logging ───────────────────────────────────────────────────────
        if self.state.global_step % 10 == 0:
            logger.info(
                f"[step {self.state.global_step}] "
                f"ce={ce_loss.item():.4f}  "
                f"ib={ib_loss.item():.4f}  "
                f"l1={l1_loss.item():.4f}  "
                f"total={total_loss.item():.4f}"
            )

        return (total_loss, outputs) if return_outputs else total_loss


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: Monitoring Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_W_sparsity(W: torch.Tensor, threshold: float = 0.1) -> float:
    """Return fraction of W entries below *threshold* (near-zero = sparse).

    Uses softplus-transformed W to match the actual modulation values.
    """
    with torch.no_grad():
        W_pos = F.softplus(W.float())
        return float((W_pos < threshold).float().mean().item())


def log_W_statistics(
    W: torch.Tensor,
    column_names: tp.List[str],
    step: int,
    save_heatmap: bool = False,
    heatmap_dir: str = ".",
) -> None:
    """Log W matrix statistics and optionally save a heatmap PNG.

    Args:
        W: Raw (un-transformed) W parameter tensor.
        column_names: Feature names for axis labels.
        step: Training step (used in log messages and file names).
        save_heatmap: If True, save a matplotlib heatmap to heatmap_dir.
        heatmap_dir: Directory for heatmap PNG files.
    """
    with torch.no_grad():
        W_pos = F.softplus(W.float()).cpu().numpy()
        sparsity = compute_W_sparsity(W)
        density = 1.0 - sparsity

        logger.info(
            f"[W monitor @ step {step}]  "
            f"density={density:.3f}  "
            f"max={W_pos.max():.4f}  "
            f"min={W_pos.min():.4f}  "
            f"mean={W_pos.mean():.4f}"
        )

        # Top-5 strongest feature interactions
        m = W_pos.shape[0]
        flat_idx = W_pos.flatten().argsort()[::-1][:5]
        for rank, idx in enumerate(flat_idx):
            i, j = divmod(int(idx), m)
            i_name = column_names[i] if i < len(column_names) else str(i)
            j_name = column_names[j] if j < len(column_names) else str(j)
            logger.info(
                f"  Top-{rank+1}: W[{i_name} → {j_name}] = {W_pos[i, j]:.4f}"
            )

        if save_heatmap:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(max(6, m), max(5, m)))
                im = ax.imshow(W_pos, aspect="auto", cmap="viridis")
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(m))
                ax.set_yticks(range(m))
                if column_names:
                    ax.set_xticklabels(column_names[:m], rotation=45, ha="right")
                    ax.set_yticklabels(column_names[:m])
                ax.set_title(f"W feature dependency matrix (step {step})")
                os.makedirs(heatmap_dir, exist_ok=True)
                path = os.path.join(heatmap_dir, f"W_heatmap_step{step:06d}.png")
                plt.tight_layout()
                plt.savefig(path, dpi=120)
                plt.close(fig)
                logger.info(f"W heatmap saved → {path}")
            except ImportError:
                logger.warning("matplotlib not installed; heatmap not saved.")


class WMonitorCallback(TrainerCallback):
    """TrainerCallback that logs W statistics at the end of each epoch."""

    def __init__(
        self,
        W: nn.Parameter,
        column_names: tp.List[str],
        save_heatmap: bool = False,
        heatmap_dir: str = ".",
    ):
        self.W = W
        self.column_names = column_names
        self.save_heatmap = save_heatmap
        self.heatmap_dir = heatmap_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        log_W_statistics(
            self.W,
            self.column_names,
            step=state.global_step,
            save_heatmap=self.save_heatmap,
            heatmap_dir=self.heatmap_dir,
        )
        return control


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6: IBSparseGReaT — 3-Phase Training Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IBSparseGReaT(GReaT):
    """GReaT extension implementing 3-phase IB-SparseAttention training.

    Phase 1 — Structure Emergence (random feature order):
        Train LoRA + W simultaneously with IB loss enabled.
        W learns which features statistically depend on each other.

    Phase 2 — Autonomous Ordering (Entropy-Directed):
        Extract W, apply bimodal gap thresholding to identify active edges.
        Apply cardinality-based asymmetric masking: if nunique[i] >> nunique[j],
        suppress the reverse edge j→i (high entropy → low entropy is the only
        physically valid causal direction).  Condense SCCs → topological sort.

    Phase 3 — Structured Fine-Tuning (fixed feature order):
        Freeze W.  Fine-tune LoRA only with optimal_order fixed.
        The sparse dependency structure guides generation.

    Args:
        All GReaT arguments plus:
        beta_ib: IB loss weight β.
        lambda_sparse: L1 sparsity weight λ for W.
        w_lr: Learning rate for the W matrix (separate from LoRA lr).
        phase1_epochs: Epochs for Phase 1.
        phase3_epochs: Epochs for Phase 3 fine-tuning (Phase 2 is analysis-only).
        cardinality_ratio_threshold: Ratio of nunique values above which the
            direction of a bidirectional edge is enforced.  When
            nunique[i] / nunique[j] > ratio, the reverse edge j→i is suppressed
            (high-entropy → low-entropy is the only valid causal direction).
            Default 10.0.  Set to inf to disable.
        save_heatmaps: Whether to save W heatmap PNGs during training.
        quantization: Base model quantisation level.  One of:
            - None   : full precision (default)
            - "8bit" : bitsandbytes LLM.int8() quantisation
            - "4bit" : bitsandbytes NF4 4-bit quantisation (bnb_4bit_quant_type="nf4",
                       double quant enabled, compute dtype bfloat16)
            Requires ``bitsandbytes`` installed (``pip install bitsandbytes``).
            W matrix is always kept in fp32 regardless of this setting.
    """

    def __init__(
        self,
        llm: str,
        *,
        # ── Plug-in modules (v0.2) ───────────────────────────────────────────
        # atomizer: CategoricalAtomizer instance or None.
        # When set, multi-token categorical values are replaced with single
        # atomic special tokens before training, eliminating token-prefix
        # competition (e.g. "Northern Cardinal" vs "Northern Mockingbird").
        # Pass atomizer=None (default) to reproduce vanilla GReaT behaviour
        # and use as ablation baseline.
        atomizer=None,   # atomic_tokenizer.CategoricalAtomizer | None
        # ── IB hyper-parameters ──────────────────────────────────────────────
        beta_ib: float = 0.1,
        lambda_sparse: float = 1e-3,
        w_lr: float = 1e-2,
        phase1_epochs: int = 10,
        phase3_epochs: int = 5,
        cardinality_ratio_threshold: float = 10.0,
        save_heatmaps: bool = False,
        heatmap_dir: str = "W_heatmaps",
        quantization: tp.Optional[str] = None,   # None | "4bit" | "8bit"
        # Pass through to GReaT
        experiment_dir: str = "trainer_ib_sparse",
        epochs: int = 10,     # used as phase1_epochs if phase1_epochs not set
        batch_size: int = 8,
        efficient_finetuning: str = "",
        lora_config: tp.Optional[tp.Dict] = None,
        float_precision: tp.Optional[int] = None,
        report_to: tp.List[str] = [],
        **train_kwargs,
    ):
        # super().__init__() loads the model in fp32; we optionally reload below
        super().__init__(
            llm=llm,
            experiment_dir=experiment_dir,
            epochs=phase1_epochs if phase1_epochs else epochs,
            batch_size=batch_size,
            efficient_finetuning="",   # LoRA applied after quantisation below
            lora_config=lora_config,
            float_precision=float_precision,
            report_to=report_to,
            **train_kwargs,
        )

        # ── Optional: reload model with bitsandbytes quantisation ─────────
        if quantization in ("4bit", "8bit"):
            self.model = self._load_quantized(llm, quantization)
            logger.info(f"Model reloaded with {quantization} quantisation.")
            # Re-apply LoRA on top of the quantised model if requested
            if efficient_finetuning == "lora":
                self._apply_lora(lora_config or {})
        elif efficient_finetuning == "lora":
            # LoRA was skipped above; apply now
            self._apply_lora(lora_config or {})
        self.beta_ib = beta_ib
        self.lambda_sparse = lambda_sparse
        self.w_lr = w_lr
        self.phase1_epochs = phase1_epochs
        self.phase3_epochs = phase3_epochs
        self.cardinality_ratio_threshold = cardinality_ratio_threshold
        self.save_heatmaps = save_heatmaps
        self.heatmap_dir = heatmap_dir

        # W is initialised lazily in fit() when num_features is known
        self.W: tp.Optional[nn.Parameter] = None
        self.modulator: tp.Optional[IBSparseAttentionModulator] = None
        self.ib_loss_computer = IBLossComputer(beta=beta_ib)

        # Populated after Phase 2
        self.optimal_order: tp.Optional[tp.List[int]] = None
        # Populated in fit() — used by Phase 2 for direction enforcement
        self.nunique_dict: tp.Optional[tp.Dict[str, int]] = None

        # ── Plug-in modules (v0.2) ───────────────────────────────────────────
        self.atomizer = atomizer   # CategoricalAtomizer | None

    # ── Quantised model loading ───────────────────────────────────────────

    @staticmethod
    def _load_quantized(llm: str, quantization: str) -> nn.Module:
        """Load the base LLM with bitsandbytes quantisation.

        Args:
            llm: HuggingFace model checkpoint.
            quantization: "4bit" or "8bit".

        Returns:
            Quantised AutoModelForCausalLM (frozen base weights, fp32 W separate).

        Raises:
            ImportError: if bitsandbytes is not installed.
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "bitsandbytes quantisation requires 'bitsandbytes' and a recent "
                "transformers.  Install with:  pip install bitsandbytes"
            )

        # Determine best compute dtype: prefer bf16 on sm_80+ (Ampere/Hopper/Blackwell)
        # bf16 has fp32-equivalent dynamic range, avoids loss-scaling, and has native
        # Tensor Core support on RTX 3090 / A100 / H100 / RTX 5080 and newer.
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _compute_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            _compute_dtype = torch.bfloat16  # CPU/MPS default

        if quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # int8 uses bf16/fp16 only for non-quantised layers
                llm_int8_threshold=6.0,
            )
        elif quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality
                bnb_4bit_use_double_quant=True,     # nested quant saves ~0.4 bpw
                bnb_4bit_compute_dtype=_compute_dtype,  # bf16 on sm_80+, fp16 otherwise
            )
        else:
            raise ValueError(f"Unknown quantization level: {quantization!r}")

        model = AutoModelForCausalLM.from_pretrained(llm, quantization_config=bnb_config)
        logger.info(
            f"Loaded {llm} with {quantization} quantisation. "
            f"W matrix will remain fp32 (separate parameter)."
        )
        return model

    # ── W initialisation ──────────────────────────────────────────────────

    def _init_W(self, num_features: int) -> None:
        """Initialise W ∈ R^{m×m} in fp32 on the model's device.

        Off-diagonal entries start at 0 (softplus(0) ≈ 0.693 → small suppression
        bias) and are shaped by IB + L1 gradients during Phase 1.

        Diagonal entries W[i,i] start at 3.0 (softplus(3) ≈ 3.05 → positive
        attention bias) so features can always attend to their own preceding
        structural tokens (e.g., "lat is" when generating the lat value).
        The diagonal is exempt from L1 regularisation (see compute_loss).
        """
        device = next(self.model.parameters()).device
        W_data = torch.zeros(num_features, num_features, dtype=torch.float32, device=device)
        # Small random perturbation for symmetry breaking (off-diagonal only)
        W_data += torch.randn_like(W_data) * 0.01
        # Diagonal: initialise high so within-feature attention is open from the start
        W_data.fill_diagonal_(3.0)
        self.W = nn.Parameter(W_data, requires_grad=True)
        self.modulator = IBSparseAttentionModulator(
            W=self.W,
            num_features=num_features,
        )
        logger.info(f"W matrix initialised: shape {self.W.shape}, device {device}")

    # ── Build optimiser with separate W learning rate ─────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with separate param groups for LoRA and W."""
        lora_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        ]
        other_model_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "lora" not in n.lower()
        ]

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": 1e-4})
        if other_model_params:
            param_groups.append({"params": other_model_params, "lr": 1e-4})
        if self.W is not None:
            param_groups.append({
                "params": [self.W],
                "lr": self.w_lr,
                "weight_decay": 0.0,   # L1 regularisation is handled in loss
            })

        return torch.optim.AdamW(param_groups, eps=1e-8)

    # ── Phase 2: Entropy-directed graph analysis & topological sort ──────

    def _derive_optimal_order(self) -> tp.List[int]:
        """Extract feature ordering from W using entropy-directed Phase 2.

        Algorithm
        ---------
        1. Bimodal gap detection — find the natural threshold separating
           suppressed edges (~0) from active edges (~25) in the W matrix.
        2. Cardinality-based asymmetric masking — for every active
           bidirectional pair (i, j):
             - if nunique[i] / nunique[j] > cardinality_ratio_threshold,
               the physical direction MUST be i→j (high entropy → low
               entropy), so the reverse edge j→i is zeroed out.
             - if the ratio is < 1/threshold, the direction is j→i instead.
             - if cardinalities are similar, both directions are kept and
               the cycle is resolved by SCC condensation.
        3. Build a directed graph with the masked active edges.
        4. Strongly Connected Component (SCC) condensation — contract cycles
           into super-nodes to form a DAG.
        5. Topological sort of the condensed DAG.
        6. Flatten: for features within the same SCC (ambiguous direction),
           sort internally by decreasing cardinality (higher entropy first).

        Returns
        -------
        List of feature indices in optimal generation order.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for Phase 2. Install with: pip install networkx"
            )

        assert self.W is not None, "W must be initialised before Phase 2."

        col_names: tp.List[str] = list(self.columns or [])
        with torch.no_grad():
            W_pos = F.softplus(self.W.float()).cpu().numpy()
        m = W_pos.shape[0]

        # ── Step 1: Bimodal gap threshold ──────────────────────────────────
        off_diag = np.array(
            [W_pos[i, j] for i in range(m) for j in range(m) if i != j]
        )
        sorted_vals = np.sort(off_diag)
        gaps = np.diff(sorted_vals)
        split_idx = int(np.argmax(gaps))
        threshold = float(
            (sorted_vals[split_idx] + sorted_vals[split_idx + 1]) / 2.0
        )
        n_active = int((off_diag > threshold).sum())
        logger.info(
            f"Phase 2: bimodal threshold={threshold:.4f}, "
            f"active edges={n_active}/{len(off_diag)}"
        )

        # ── Step 2: Cardinality-based asymmetric masking ───────────────────
        W_masked = W_pos.copy()
        nunique: tp.Optional[np.ndarray] = None

        if (
            self.nunique_dict is not None
            and len(self.nunique_dict) == m
            and self.cardinality_ratio_threshold < float("inf")
        ):
            nunique = np.array(
                [self.nunique_dict.get(c, 1) for c in col_names], dtype=float
            )
            ratio_thresh = self.cardinality_ratio_threshold
            cuts = 0

            for i in range(m):
                for j in range(i + 1, m):
                    # Only examine pairs where at least one direction is active
                    if W_pos[i, j] <= threshold and W_pos[j, i] <= threshold:
                        continue

                    ratio = nunique[i] / (nunique[j] + 1e-8)

                    if ratio > ratio_thresh:
                        # nunique[i] >> nunique[j]: physical direction i→j
                        # Suppress reverse edge j→i
                        if W_masked[j, i] > threshold:
                            W_masked[j, i] = 0.0
                            cuts += 1
                            logger.info(
                                f"  [cardinality cut] {col_names[j]}->{col_names[i]} "
                                f"suppressed  "
                                f"(nunique {col_names[i]}={nunique[i]:.0f} >> "
                                f"{col_names[j]}={nunique[j]:.0f}, "
                                f"ratio={ratio:.1f})"
                            )
                    elif ratio < 1.0 / ratio_thresh:
                        # nunique[j] >> nunique[i]: physical direction j→i
                        # Suppress forward edge i→j
                        if W_masked[i, j] > threshold:
                            W_masked[i, j] = 0.0
                            cuts += 1
                            logger.info(
                                f"  [cardinality cut] {col_names[i]}->{col_names[j]} "
                                f"suppressed  "
                                f"(nunique {col_names[j]}={nunique[j]:.0f} >> "
                                f"{col_names[i]}={nunique[i]:.0f}, "
                                f"ratio={1/ratio:.1f})"
                            )
                    # else: similar cardinality — keep both, resolve via SCC

            logger.info(
                f"Phase 2: cardinality masking suppressed {cuts} reverse edges"
            )
        else:
            logger.info(
                "Phase 2: cardinality masking skipped "
                "(nunique_dict unavailable or ratio_threshold=inf)"
            )

        # ── Step 3: Build directed graph with masked active edges ──────────
        G = nx.DiGraph()
        G.add_nodes_from(range(m))
        for i in range(m):
            for j in range(m):
                if i != j and W_masked[i, j] > threshold:
                    G.add_edge(i, j, weight=float(W_masked[i, j]))

        log_W_statistics(self.W, col_names, step=-1)
        logger.info(
            f"Phase 2: graph has {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges after cardinality masking"
        )

        # ── Step 4: SCC condensation → DAG ────────────────────────────────
        # Find SCCs; within each SCC sort by decreasing cardinality so that
        # the highest-entropy feature is listed first inside the group.
        raw_sccs = list(nx.strongly_connected_components(G))

        def _sort_scc(scc: tp.Set[int]) -> tp.List[int]:
            nodes = list(scc)
            if nunique is not None and len(nodes) > 1:
                nodes.sort(key=lambda n: -nunique[n])
            return nodes

        sorted_sccs = [_sort_scc(s) for s in raw_sccs]

        # nx.condensation requires scc as a list of sets
        scc_sets = [set(s) for s in sorted_sccs]
        C = nx.condensation(G, scc=scc_sets)
        # C.nodes[k]['members'] is the set of original nodes in SCC k

        # ── Step 5: Topological sort of condensed DAG ─────────────────────
        try:
            topo_scc_ids = list(nx.topological_sort(C))
        except nx.exception.NetworkXUnfeasible:
            logger.warning(
                "Phase 2: topological sort on condensed DAG failed "
                "(should be impossible); using SCC discovery order."
            )
            topo_scc_ids = list(range(len(raw_sccs)))

        # ── Step 6: Flatten SCCs → feature ordering ────────────────────────
        # Each SCC in topo order; members already sorted by decreasing nunique.
        optimal_order: tp.List[int] = []
        for scc_id in topo_scc_ids:
            members = sorted_sccs[scc_id]
            optimal_order.extend(members)

        # Safety: append any node missed due to graph isolation
        in_order = set(optimal_order)
        missing = [i for i in range(m) if i not in in_order]
        if missing:
            if nunique is not None:
                missing.sort(key=lambda n: -nunique[n])
            optimal_order.extend(missing)

        logger.info(
            f"Phase 2: optimal order = {optimal_order}\n"
            "  Features: "
            + " -> ".join(col_names[i] for i in optimal_order if i < len(col_names))
        )
        return optimal_order

    # ── Main fit() with 3-phase logic ─────────────────────────────────────

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
        random_conditional_col: bool = False,   # disabled: IB controls ordering
    ) -> tp.Dict[str, tp.Any]:
        """3-Phase IB-SparseAttention training.

        Args:
            data: Tabular training data (DataFrame or ndarray).
            column_names: Column names (required if data is ndarray).
            conditional_col: Conditional column for start sampling.
            resume_from_checkpoint: Resume Phase 1 from checkpoint path.
            random_conditional_col: Ignored (IB controls ordering).

        Returns:
            Dict with keys:
              'phase1_trainer': IBSparseTrainer from Phase 1
              'phase3_trainer': IBSparseTrainer from Phase 3
              'optimal_order': List[int] — feature ordering from Phase 2
              'W': Final W parameter tensor (detached, cpu)
        """
        df = _array_to_dataframe(data, columns=column_names)

        # ── Atomic tokenisation plug-in (v0.2) ────────────────────────────────
        # Must run BEFORE _update_column_information so GReaT's internal
        # column-value metadata reflects the atomised representation.
        # After fit_transform the tokenizer has new special tokens; the model
        # embedding matrix is resized here so all three phases see consistent
        # dimensions.
        if self.atomizer is not None:
            df = self.atomizer.fit_transform(df, self.tokenizer)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(
                f"[AtomicTokenizer] vocab expanded to {len(self.tokenizer)} "
                f"(+{self.atomizer.n_new_tokens} atomic tokens)"
            )

        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        m = len(self.columns)
        logger.info(f"IBSparseGReaT.fit(): {m} features, {len(df)} samples")

        # ── Compute cardinality (nunique) for Phase 2 direction enforcement ──
        self.nunique_dict = {col: int(df[col].nunique()) for col in df.columns}
        logger.info(
            "Feature cardinality: "
            + ", ".join(f"{c}={v}" for c, v in self.nunique_dict.items())
        )

        # ── Initialise W matrix ───────────────────────────────────────────
        self._init_W(m)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Structure Emergence
        # ═══════════════════════════════════════════════════════════════════
        logger.info("═══ Phase 1: Structure Emergence ═══")
        phase1_trainer = self._run_phase(
            df=df,
            phase_name="phase1",
            num_epochs=self.phase1_epochs,
            fixed_order=None,        # random feature order
            freeze_W=False,          # train W + LoRA together
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Autonomous Ordering (analysis only — no gradient update)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("═══ Phase 2: Autonomous Ordering ═══")
        self.optimal_order = self._derive_optimal_order()
        logger.info(f"Optimal feature order: {self.optimal_order}")
        logger.info(
            "Features in order: "
            + " → ".join(self.columns[i] for i in self.optimal_order if i < len(self.columns))
        )
        W_density = 1.0 - compute_W_sparsity(self.W)
        logger.info(f"W density after Phase 1: {W_density:.3f}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Structured Fine-Tuning
        # ═══════════════════════════════════════════════════════════════════
        logger.info("═══ Phase 3: Structured Fine-Tuning ═══")
        phase3_trainer = self._run_phase(
            df=df,
            phase_name="phase3",
            num_epochs=self.phase3_epochs,
            fixed_order=self.optimal_order,  # lock in optimal order
            freeze_W=True,                   # only LoRA trained
            resume_from_checkpoint=False,
        )

        logger.info("IBSparseGReaT training complete.")
        return {
            "phase1_trainer": phase1_trainer,
            "phase3_trainer": phase3_trainer,
            "optimal_order": self.optimal_order,
            "W": self.W.detach().cpu(),
        }

    # ── Shared phase runner ───────────────────────────────────────────────

    def _run_phase(
        self,
        df: pd.DataFrame,
        phase_name: str,
        num_epochs: int,
        fixed_order: tp.Optional[tp.List[int]],
        freeze_W: bool,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> IBSparseTrainer:
        """Build and run one training phase.

        Args:
            df: Training DataFrame.
            phase_name: Subdirectory name under experiment_dir.
            num_epochs: Number of training epochs.
            fixed_order: Feature index order (None = random shuffle).
            freeze_W: If True, W is not updated (gradient disabled).
            resume_from_checkpoint: Checkpoint path or False.

        Returns:
            The IBSparseTrainer instance after training.
        """
        phase_dir = os.path.join(self.experiment_dir, phase_name)
        m = len(self.columns)

        # ── Dataset ───────────────────────────────────────────────────────
        great_ds = FeatureAlignmentDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, self.float_precision)
        great_ds.set_ib_config(num_features=m, fixed_order=fixed_order)

        # ── W gradient control ────────────────────────────────────────────
        self.W.requires_grad_(not freeze_W)

        # ── Training arguments ────────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=phase_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.batch_size,
            remove_unused_columns=False,
            **self.train_hyperparameters,
        )

        # ── Callbacks ─────────────────────────────────────────────────────
        callbacks = [
            WMonitorCallback(
                W=self.W,
                column_names=self.columns or [],
                save_heatmap=self.save_heatmaps,
                heatmap_dir=os.path.join(self.heatmap_dir, phase_name),
            )
        ]

        # ── Trainer ───────────────────────────────────────────────────────
        trainer = IBSparseTrainer(
            model=self.model,
            args=training_args,
            train_dataset=great_ds,
            processing_class=self.tokenizer,
            data_collator=FeatureAlignmentCollator(self.tokenizer),
            callbacks=callbacks,
            # IB-specific
            modulator=self.modulator,
            ib_loss_computer=self.ib_loss_computer,
            W=self.W,
            lambda_sparse=self.lambda_sparse,
            freeze_W=freeze_W,
        )

        # Override the default optimiser to give W its own learning rate
        trainer.optimizer = self._build_optimizer()

        # ── Run ───────────────────────────────────────────────────────────
        logger.info(
            f"Starting {phase_name}: {num_epochs} epochs, "
            f"fixed_order={fixed_order is not None}, freeze_W={freeze_W}"
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return trainer

    # ── Sample override (uses optimal_order if available) ─────────────────

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist=None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cuda",
        guided_sampling: bool = False,
        random_feature_order: bool = True,
        conditions: tp.Optional[tp.Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Generate samples.  If Phase 2 has been run, use optimal_order."""
        # Capture original column order before any temporary reordering.
        # Initialised to None so the finally block is a no-op when optimal_order
        # is not set (avoids unbound-variable risk).
        original_columns = None

        if self.optimal_order is not None:
            logger.info(
                "Sampling with optimal feature order from Phase 2. "
                "Setting random_feature_order=False."
            )
            random_feature_order = False
            # Temporarily reorder self.columns so that GReaT's guided sampler
            # generates features in the IB-optimal sequence.
            if self.columns is not None:
                original_columns = list(self.columns)
                self.columns = [original_columns[i] for i in self.optimal_order]

        try:
            result = super().sample(
                n_samples=n_samples,
                start_col=start_col,
                start_col_dist=start_col_dist,
                temperature=temperature,
                k=k,
                max_length=max_length,
                drop_nan=drop_nan,
                device=device,
                guided_sampling=guided_sampling,
                random_feature_order=random_feature_order,
                conditions=conditions,
            )
        finally:
            # Always restore self.columns, even if super().sample() raises.
            # Without this, a failed generation call would leave the model in
            # an inconsistent state where the next sample() call fails.
            if original_columns is not None:
                self.columns = original_columns

        # Restore result DataFrame column order to match training data layout.
        if original_columns is not None:
            try:
                result = result[original_columns]
            except KeyError:
                pass

        # ── Inverse atomic tokenisation plug-in (v0.2) ───────────────────────
        # Replace atomic token strings (e.g. ATM_bird_3) back to original
        # categorical values (e.g. "California Quail") in the output DataFrame.
        # Unknown/hallucinated atomic tokens are left as-is for inspection.
        if self.atomizer is not None:
            result = self.atomizer.inverse_transform(result)

        return result

    # ── Convenience: access W as a readable dependency matrix ─────────────

    def get_dependency_matrix(self) -> pd.DataFrame:
        """Return the learned feature dependency matrix as a labelled DataFrame.

        Returns softplus-transformed W (actual modulation values, not raw params).
        """
        if self.W is None:
            raise ValueError("W not yet initialised — call fit() first.")
        with torch.no_grad():
            W_pos = F.softplus(self.W.float()).cpu().numpy()
        cols = self.columns or [str(i) for i in range(W_pos.shape[0])]
        return pd.DataFrame(W_pos, index=cols, columns=cols)
