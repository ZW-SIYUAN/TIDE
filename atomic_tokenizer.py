# -*- coding: utf-8 -*-
"""
atomic_tokenizer.py  --  Atomic Categorical Token Plug-in for TIDE
==================================================================

Update v0.2 | 2026-03-01
-------------------------
Why this update
~~~~~~~~~~~~~~~
In the US-Location experiment the synthetic data completely missed 13 of 51
states (CA, CT, DC, DE, HI, IN, MD, ME, MI, NV, OH, PA, VA), despite each
state having ~320 training samples.

Root-cause analysis identified two distinct failure modes:

  Failure Mode A -- Geographic (CA, NV):
    The IB-optimal generation order (lon -> lat -> state_code -> bird ->
    lat_zone) causes P_synth(lat | lon=-120) to peak at WA/OR latitudes
    (z > 0.8).  The CA/NV latitude range (z < 0.5) is never sampled even
    though the Mahalanobis distance is < 2sigma.  This is a marginal-
    distribution narrowing issue addressed by temperature and/or order
    adjustments in the experiment configuration.

  Failure Mode B -- Token competition (OH, IN, PA, VA, CT, MI, and 7 more):
    "Northern Cardinal" (8 states, 2551 training rows) generated only 7
    synthetic rows.  "American Robin" (3 states, 969 training rows) generated
    only 1 row.  Both share GPT-2 token prefixes with dominant birds:

        "Northern " -> always "Mockingbird" (2524 synth) never "Cardinal" (7)
        "American " -> rarely  "Robin"      (1 synth)

    GReaT encodes every categorical value as a raw text substring.  When two
    values share a GPT-2 prefix token, the more-frequent suffix dominates all
    probability mass regardless of the conditioning context.  This is a
    structural flaw in text-based tabular generation: surface-form similarity
    of column values interferes with their statistical independence.

Fix (this module)
~~~~~~~~~~~~~~~~~
Register every multi-token categorical value as a *single* atomic special
token.  After atomisation "Northern Cardinal" -> ATM_bird_3 and "Northern
Mockingbird" -> ATM_bird_8.  Both are now independent vocabulary entries that
compete purely on their context-conditioned logits, not on shared prefixes.

Design principles
~~~~~~~~~~~~~~~~~
  1. Plug-and-play: zero changes to ib_sparse_attention.py internals beyond
     accepting ``atomizer=CategoricalAtomizer()`` in IBSparseGReaT.__init__().
  2. Ablation-safe: pass ``atomizer=None`` (default) to restore vanilla GReaT.
  3. Selective: only values that tokenise to >= min_tokens are atomised.
     Single-token values are already atomic; changing them adds overhead with
     no benefit.
  4. Transparent to FeatureAlignmentDataset: each atomised value becomes
     exactly 1 token, so offset-based feature_map assignment is unaffected.
  5. Persistent: save() / load() serialise the mapping for offline analysis
     or resuming experiments.

Usage
~~~~~
    from atomic_tokenizer import CategoricalAtomizer, diagnose

    # 1. Inspect which columns benefit most
    diagnose(df_train, cat_cols=["bird", "state_code", "lat_zone"],
             tokenizer=tokenizer)

    # 2. Create and pass to model
    atomizer = CategoricalAtomizer()
    model = IBSparseGReaT(llm="gpt2-medium", atomizer=atomizer, ...)
    model.fit(df_train, conditional_col="bird")

    # 3. Synthetic data is auto-decoded back to original values
    syn_df = model.sample(n_samples=1000, ...)   # bird values = "Northern Cardinal" etc.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# Token string format: ATM_{sanitised_col}_{index}
# No angle brackets or special chars -- robust to any regex-based text parser.
_ATM_PREFIX   = "ATM"
_SANITIZE_RE  = re.compile(r"[^A-Za-z0-9]")
_ATM_PATTERN  = re.compile(r"ATM_[A-Za-z0-9_]+_\d+")   # for detection in text


def _sanitize(name: str) -> str:
    """Replace non-alphanumeric chars with underscore for use in token names."""
    return _SANITIZE_RE.sub("_", name)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic utility
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(
    df: pd.DataFrame,
    cat_cols: List[str],
    tokenizer,
    min_tokens: int = 2,
) -> pd.DataFrame:
    """
    Report token-competition risk for each categorical column.

    Prints and returns a DataFrame with per-column statistics:
      - n_values         : number of unique values
      - n_need_atomize   : values with >= min_tokens sub-tokens
      - prefix_overlap   : fraction sharing the first sub-token with another value
      - recommendation   : "atomize" | "skip"

    High prefix_overlap + high n_need_atomize => high competition risk.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list of column names to analyse
    tokenizer : HuggingFace tokenizer (already loaded)
    min_tokens : threshold; default 2

    Returns
    -------
    pd.DataFrame  summary table
    """
    rows = []
    for col in cat_cols:
        if col not in df.columns:
            log.warning(f"  diagnose: column '{col}' not in DataFrame, skipped.")
            continue

        vals = df[col].dropna().unique().astype(str).tolist()
        n_vals = len(vals)

        # Sub-token counts and first tokens
        tok_seqs   = [tokenizer.encode(v) for v in vals]
        n_multi    = sum(1 for s in tok_seqs if len(s) >= min_tokens)
        first_toks = [s[0] for s in tok_seqs]
        c          = Counter(first_toks)
        n_shared   = sum(cnt for cnt in c.values() if cnt > 1)
        overlap    = n_shared / n_vals if n_vals else 0.0
        recommend  = "atomize" if n_multi > 0 else "skip"

        rows.append({
            "column":          col,
            "n_values":        n_vals,
            "n_need_atomize":  n_multi,
            "prefix_overlap":  round(overlap, 3),
            "recommendation":  recommend,
        })
        log.info(
            f"  [{col:15s}]  {n_vals:3d} values  "
            f"atomize={n_multi:3d}  overlap={overlap:.1%}  -> {recommend}"
        )

    summary = pd.DataFrame(rows)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalAtomizer:
    """
    Eliminates token-prefix competition among categorical values by
    registering each multi-token value as a single atomic special token.

    Lifecycle
    ---------
    1. atomizer = CategoricalAtomizer()

    2. model = IBSparseGReaT(llm=..., atomizer=atomizer, ...)

    3. model.fit(df, ...)
         Internally calls:
           atomizer.fit_transform(df, tokenizer)  -- in IBSparseGReaT.fit()
           model.resize_token_embeddings(len(tokenizer))

    4. syn_df = model.sample(...)
         Internally calls:
           atomizer.inverse_transform(syn_df_raw)  -- in IBSparseGReaT.sample()

    Parameters
    ----------
    cat_cols : list[str] | None
        Columns to consider.  None = auto-detect (dtype == object).
    min_tokens : int
        Minimum GPT-2 sub-token count for a value to be atomised.  Default 2.
    verbose : bool
        Log per-column details during fit().
    """

    def __init__(
        self,
        cat_cols: Optional[List[str]] = None,
        min_tokens: int = 2,
        verbose: bool = True,
    ) -> None:
        self.cat_cols    = cat_cols
        self.min_tokens  = min_tokens
        self.verbose     = verbose

        # Filled after fit()
        self._v2a: Dict[str, Dict[str, str]] = {}   # col -> {orig_val -> atm_tok}
        self._a2v: Dict[str, Dict[str, str]] = {}   # col -> {atm_tok -> orig_val}
        self._new_tokens: List[str] = []            # flat list for tokenizer
        self.fitted = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        tokenizer,
        cat_cols: Optional[List[str]] = None,
    ) -> "CategoricalAtomizer":
        """
        Scan df, identify multi-token values, register atomic special tokens.

        After this call the tokenizer has new entries.  The caller MUST run:
            model.resize_token_embeddings(len(tokenizer))
        before any forward pass.

        Parameters
        ----------
        df : pd.DataFrame  training data (original values, not yet transformed)
        tokenizer : HuggingFace tokenizer (modified in-place)
        cat_cols  : override self.cat_cols; None = use self.cat_cols or auto-detect
        """
        self._v2a.clear()
        self._a2v.clear()
        self._new_tokens.clear()

        cols = cat_cols or self.cat_cols or [
            c for c in df.columns if df[c].dtype == object
        ]

        if self.verbose:
            log.info("CategoricalAtomizer.fit() -- scanning columns ...")
            log.info("  Token-competition diagnostic:")
            diagnose(df, cols, tokenizer, self.min_tokens)

        new_special: List[str] = []

        for col in cols:
            if col not in df.columns:
                continue

            vals = sorted(df[col].dropna().unique().astype(str).tolist())
            v2a: Dict[str, str] = {}
            a2v: Dict[str, str] = {}
            n_atomised = 0

            for idx, val in enumerate(vals):
                n_tok = len(tokenizer.encode(val))
                if n_tok >= self.min_tokens:
                    col_safe = _sanitize(col)
                    atm_tok  = f"{_ATM_PREFIX}_{col_safe}_{idx}"
                    v2a[val]     = atm_tok
                    a2v[atm_tok] = val
                    new_special.append(atm_tok)
                    n_atomised += 1

            if n_atomised:
                self._v2a[col] = v2a
                self._a2v[col] = a2v
                if self.verbose:
                    log.info(
                        f"  [{col}]  {n_atomised}/{len(vals)} values atomised "
                        f"(>= {self.min_tokens} sub-tokens)"
                    )

        if new_special:
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": new_special}
            )
            self._new_tokens = new_special
            log.info(
                f"CategoricalAtomizer: registered {added} atomic tokens "
                f"-- new vocab size = {len(tokenizer)}"
            )
            log.info(
                "  ACTION REQUIRED: call model.resize_token_embeddings("
                f"{len(tokenizer)}) before training."
            )
        else:
            log.info("CategoricalAtomizer: no values required atomisation.")

        self.fitted = True
        return self

    # ── transform / inverse_transform ─────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace multi-token categorical values with atomic token strings."""
        self._require_fitted()
        df = df.copy()
        for col, v2a in self._v2a.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).map(lambda v, _m=v2a: _m.get(v, v))
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restore atomic token strings to original categorical values.

        Unknown atomic tokens (model hallucinations) are left as-is so the
        caller can decide whether to drop_nan or keep them.
        """
        self._require_fitted()
        df = df.copy()
        for col, a2v in self._a2v.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).map(lambda v, _m=a2v: _m.get(v, v))
        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        tokenizer,
        cat_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        return self.fit(df, tokenizer, cat_cols).transform(df)

    # ── utilities ─────────────────────────────────────────────────────────────

    @property
    def n_new_tokens(self) -> int:
        """Number of atomic tokens registered with the tokenizer."""
        return len(self._new_tokens)

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame of all (column, original_value, atomic_token) triples.

        Useful for inspection and for saving the full mapping to CSV.
        """
        rows = []
        for col, v2a in self._v2a.items():
            for orig, atm in v2a.items():
                rows.append({"column": col, "original_value": orig, "atomic_token": atm})
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """
        Persist the mapping tables to a JSON file.

        Note: the tokenizer's special-token list must also be saved separately
        (the standard tokenizer.save_pretrained() handles that).
        """
        payload = {
            "min_tokens": self.min_tokens,
            "v2a": self._v2a,
            "a2v": self._a2v,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"CategoricalAtomizer saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "CategoricalAtomizer":
        """
        Restore from a JSON file.

        The tokenizer must be re-fitted (loaded with save_pretrained) so that
        the special tokens are present; call model.resize_token_embeddings()
        after loading.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(min_tokens=payload["min_tokens"])
        obj._v2a = payload["v2a"]
        obj._a2v = payload["a2v"]
        obj._new_tokens = [
            tok for col_map in payload["v2a"].values() for tok in col_map.values()
        ]
        obj.fitted = True
        log.info(f"CategoricalAtomizer loaded <- {path}  ({obj.n_new_tokens} tokens)")
        return obj

    # ── private ───────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                "CategoricalAtomizer has not been fitted. "
                "Call .fit() or .fit_transform() first."
            )

    def __repr__(self) -> str:
        cols = list(self._v2a.keys()) if self.fitted else "[]"
        return (
            f"CategoricalAtomizer(fitted={self.fitted}, "
            f"n_new_tokens={self.n_new_tokens}, "
            f"cols={cols})"
        )
