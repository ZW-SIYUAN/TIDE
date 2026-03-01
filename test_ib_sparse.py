"""
Smoke-test for ib_sparse_attention.py

Tests (without GPU/full training):
  1. Import all public symbols
  2. FeatureAlignmentDataset tokenises correctly and produces feature_map
  3. FeatureAlignmentCollator pads feature_map to uniform length
  4. IBSparseAttentionModulator computes attention bias with correct shape
  5. IBLossComputer returns a differentiable scalar
  6. IBSparseGReaT can be instantiated (does NOT run full training)

Run: python test_ib_sparse.py
"""
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── Helper: coloured output ─────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}PASS{RESET} {msg}")
def fail(msg, exc=None):
    print(f"  {RED}FAIL{RESET} {msg}")
    if exc:
        traceback.print_exc()
    sys.exit(1)

# ─── 1. Import ────────────────────────────────────────────────────────────────
print("\n[1] Importing ib_sparse_attention …")
try:
    from ib_sparse_attention import (
        FeatureAlignmentDataset,
        FeatureAlignmentCollator,
        IBSparseAttentionModulator,
        IBLossComputer,
        IBSparseTrainer,
        IBSparseGReaT,
        compute_W_sparsity,
        log_W_statistics,
        WMonitorCallback,
    )
    ok("All symbols imported successfully")
except Exception as e:
    fail("Import failed", e)

# ─── 2. FeatureAlignmentDataset tokenisation ──────────────────────────────────
print("\n[2] FeatureAlignmentDataset …")
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    df = pd.DataFrame({
        "age":    [25, 30],
        "income": [50000, 80000],
        "city":   ["Paris", "London"],
    })

    ds = FeatureAlignmentDataset.from_pandas(df)
    ds.set_tokenizer(tokenizer)
    ds.set_ib_config(num_features=3)

    item = ds._getitem(0)
    assert "input_ids" in item, "missing input_ids"
    assert "feature_map" in item, "missing feature_map"
    assert len(item["input_ids"]) == len(item["feature_map"]), (
        f"length mismatch: ids={len(item['input_ids'])} "
        f"map={len(item['feature_map'])}"
    )
    # At least some tokens must have a feature index ≥ 0
    assert any(f >= 0 for f in item["feature_map"]), "all feature indices are -1"
    ok(f"item produced: {len(item['input_ids'])} tokens, "
       f"feature_map unique values = {sorted(set(item['feature_map']))}")
except Exception as e:
    fail("FeatureAlignmentDataset failed", e)

# ─── 3. FeatureAlignmentCollator ─────────────────────────────────────────────
print("\n[3] FeatureAlignmentCollator …")
try:
    collator = FeatureAlignmentCollator(tokenizer)

    # Create two fake items with different lengths
    fake_items = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "feature_map": [0, 0, 1]},
        {"input_ids": [4, 5],    "attention_mask": [1, 1],    "feature_map": [2, 2]},
    ]
    batch = collator(fake_items)

    assert "input_ids"   in batch, "missing input_ids"
    assert "feature_map" in batch, "missing feature_map"
    assert batch["input_ids"].shape[0]   == 2, "batch size wrong"
    assert batch["feature_map"].shape[0] == 2, "feature_map batch size wrong"
    assert batch["input_ids"].shape[1]   == batch["feature_map"].shape[1], (
        "feature_map not padded to input_ids length"
    )
    ok(f"batch shapes: input_ids={tuple(batch['input_ids'].shape)}, "
       f"feature_map={tuple(batch['feature_map'].shape)}")
except Exception as e:
    fail("FeatureAlignmentCollator failed", e)

# ─── 4. IBSparseAttentionModulator ────────────────────────────────────────────
print("\n[4] IBSparseAttentionModulator …")
try:
    m = 3
    W = nn.Parameter(torch.zeros(m, m))
    mod = IBSparseAttentionModulator(W=W, num_features=m)

    B, L = 2, 8
    feature_map = torch.randint(-1, m, (B, L))
    # Ensure at least some valid feature indices
    feature_map[0, 2:] = torch.randint(0, m, (6,))
    mod.set_feature_map(feature_map)

    bias = mod.compute_attention_bias(L=L, device=torch.device("cpu"), dtype=torch.float32)
    assert bias is not None, "bias is None"
    assert bias.shape == (B, 1, L, L), f"unexpected bias shape: {bias.shape}"
    ok(f"attention bias shape: {tuple(bias.shape)}, "
       f"range=[{bias.min().item():.3f}, {bias.max().item():.3f}]")
except Exception as e:
    fail("IBSparseAttentionModulator failed", e)

# ─── 5. IBLossComputer ────────────────────────────────────────────────────────
print("\n[5] IBLossComputer …")
try:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    ib_comp = IBLossComputer(beta=0.1)

    B, L, V = 2, 10, model.config.vocab_size
    # Fake logits that require grad (simulates main forward pass output)
    logits = torch.randn(B, L, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, L))
    feature_map = torch.zeros(B, L, dtype=torch.long)
    # Last 3 tokens belong to feature 2 (the "target" feature)
    feature_map[:, 7:] = 2

    loss = ib_comp.compute(
        model=model,
        logits_with_context=logits,
        input_ids=input_ids,
        feature_map=feature_map,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    assert loss.shape == (), f"loss should be scalar, got {loss.shape}"
    # Check gradient flows to logits
    loss.backward()
    assert logits.grad is not None, "no gradient through logits"
    ok(f"IB loss = {loss.item():.6f}, gradient confirmed")
except Exception as e:
    fail("IBLossComputer failed", e)

# ─── 6. IBSparseGReaT instantiation ──────────────────────────────────────────
print("\n[6] IBSparseGReaT instantiation …")
try:
    ib_great = IBSparseGReaT(
        llm="distilgpt2",
        beta_ib=0.1,
        lambda_sparse=1e-3,
        w_lr=1e-2,
        phase1_epochs=1,
        phase3_epochs=1,
        batch_size=2,
        efficient_finetuning="",  # no LoRA for quick test
        experiment_dir="test_ib_great_output",
    )
    assert ib_great.W is None, "W should be None before fit()"
    assert ib_great.optimal_order is None
    ok("IBSparseGReaT instantiated successfully")
except Exception as e:
    fail("IBSparseGReaT instantiation failed", e)

# ─── 7. W sparsity utility ────────────────────────────────────────────────────
print("\n[7] Monitoring utilities …")
try:
    W_test = nn.Parameter(torch.randn(4, 4))
    sparsity = compute_W_sparsity(W_test, threshold=0.5)
    assert 0.0 <= sparsity <= 1.0, f"sparsity out of range: {sparsity}"
    ok(f"W sparsity = {sparsity:.3f}")
except Exception as e:
    fail("Monitoring utilities failed", e)

# ─── 8. CategoricalAtomizer (v0.2 plug-in) ───────────────────────────────────
print("\n[8] CategoricalAtomizer (v0.2) …")
try:
    import tempfile, os
    from atomic_tokenizer import CategoricalAtomizer, diagnose

    # Minimal DataFrame with multi-token categorical values
    df_atm = pd.DataFrame({
        "bird":       ["Northern Cardinal", "Northern Mockingbird", "American Robin",
                       "Northern Cardinal", "American Robin"],
        "state_code": ["OH", "IN", "PA", "VA", "OH"],
        "lat":        [40.1, 39.8, 40.2, 37.5, 40.1],
    })

    atm = CategoricalAtomizer(cat_cols=["bird", "state_code"], min_tokens=2, verbose=False)
    df_transformed = atm.fit_transform(df_atm, tokenizer)

    # All bird values should be replaced by ATM_* tokens
    assert atm.fitted, "atomizer should be fitted"
    assert atm.n_new_tokens > 0, "should have registered new tokens"

    original_birds = set(df_atm["bird"].unique())
    transformed_birds = set(df_transformed["bird"].unique())
    assert not any(b in transformed_birds for b in original_birds), (
        f"original values still present after transform: {transformed_birds & original_birds}"
    )
    assert all(v.startswith("ATM_") for v in transformed_birds), (
        f"transformed values should all be ATM_* tokens: {transformed_birds}"
    )

    # Inverse transform should restore original values
    df_restored = atm.inverse_transform(df_transformed)
    assert set(df_restored["bird"].unique()) == original_birds, (
        "inverse_transform did not restore all original bird values"
    )

    # save / load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "atm.json")
        atm.save(save_path)
        atm2 = CategoricalAtomizer.load(save_path)
        df_restored2 = atm2.inverse_transform(df_transformed)
        assert list(df_restored2["bird"]) == list(df_restored["bird"]), (
            "save/load round-trip changed inverse_transform result"
        )

    ok(
        f"CategoricalAtomizer: {atm.n_new_tokens} new tokens, "
        f"transform/inverse/save/load all correct"
    )
except Exception as e:
    fail("CategoricalAtomizer failed", e)

# ─── 9. IBSparseGReaT + atomizer instantiation ────────────────────────────────
print("\n[9] IBSparseGReaT with atomizer=CategoricalAtomizer …")
try:
    from atomic_tokenizer import CategoricalAtomizer as CA
    atm9 = CA(cat_cols=["city"], verbose=False)
    ib9 = IBSparseGReaT(
        llm="distilgpt2",
        atomizer=atm9,
        beta_ib=0.1,
        phase1_epochs=1,
        phase3_epochs=1,
        batch_size=2,
        experiment_dir="test_ib_great_output_v2",
    )
    assert ib9.atomizer is atm9, "atomizer not stored correctly"
    assert not atm9.fitted, "atomizer should not be fitted yet (fit() not called)"
    ok("IBSparseGReaT(atomizer=...) instantiated; atomizer stored, not yet fitted")
except Exception as e:
    fail("IBSparseGReaT + atomizer instantiation failed", e)

print(f"\n{GREEN}All tests passed!{RESET}\n")
