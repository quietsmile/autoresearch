# AutoResearch: 8-GPU Parallel Neural Architecture Search — Progress Report II

**Date:** March 9, 2026 (09:52 UTC)
**Total Experiments:** ~1,451 across 8 GPUs
**Cumulative Duration:** ~26 hours
**Baseline val_bpb:** 0.997900
**Best val_bpb (this report):** 0.968240 (GPU 1) — **3.0% improvement over baseline**
**Previous report best:** 0.972300 (March 9, ~08:00) — **+0.4% additional gain**

---

## Abstract

Continuing from the first progress report (best: 0.9723), the 8-GPU parallel search has pushed val_bpb to **0.9682** through two independent discoveries: (1) applying Value Embedding (VE) to all transformer layers rather than alternating layers, and (2) finding that lower weight decay on VE parameters (0.01 vs default) provides a systematic improvement. Concurrently, GPU 4 discovered that reducing depth from 8 to 7 layers (while preserving model width AR=96) yields more optimizer steps within the 5-minute budget, reaching 0.9707. These findings reveal a consistent theme: within a fixed wall-clock budget, **compute efficiency per step dominates over raw model capacity**.

---

## 1. Current State

### 1.1 Per-GPU Statistics (Updated)

| GPU | Branch | Total Exps | Keep | Discard | Crash | Best val_bpb | Delta vs Baseline |
|-----|--------|-----------|------|---------|-------|-------------|-------------------|
| 0 | mar8-gpu0 | 181 | 19 | 155 | 7 | 0.9743 | −2.4% |
| 1 | mar8-gpu1 | 229 | 34 | 190 | 5 | **0.9682** 🏆 | **−3.0%** |
| 2 | mar8-gpu2 | 224 | 30 | — | — | 0.9770 | −2.7% |
| 3 | mar8-gpu3 | 209 | 27 | 179 | 3 | 0.9798 | −2.0% |
| 4 | mar8-gpu4 | 226 | 35 | 186 | 5 | 0.9707 | −2.9% |
| 5 | mar8-gpu5 | 116 | 17 | 97 | 2 | 0.9790 | −2.0% |
| 6 | mar8-gpu6 | 216 | 42 | 168 | 6 | 0.9743 | −2.4% |
| 7 | mar8-gpu7 | 43 | 8 | 33 | 2 | 0.9840 | −1.6% |
| **Total** | — | **1,444** | **212** | **~1,008** | **30** | **0.9682** | **−3.0%** |

Note: GPU 2 uses an autonomous background-agent loop; its discard/crash counts are not separately tracked.

### 1.2 Architecture Divergence

By this point, the 8 search lines have diverged into distinct architectural hypotheses:

| GPU | DEPTH | AR (model_dim) | Window Pattern | Key Exploration Focus |
|-----|-------|----------------|----------------|-----------------------|
| 0 | 8 | 96 (768) | SSSM | Muon beta2, WD tuning |
| 1 | 8 | 96 (768) | SSSM | VE on all layers, VE regularization |
| 2 | 9 | 64 (576) | SSSSL | ns_steps, x0 optimizer |
| 3 | 9 | 64 (576) | SSSL | RoPE base, ve_gate |
| 4 | **7** | 96 (768) | SSSM | Depth reduction, softmax_scale |
| 5 | 10 | 64 (640) | SSSL | x0_lambdas highway, WARMDOWN |
| 6 | **7** | 96 (768) | SSSL | DEPTH=7 + ve_gate, Adam beta |
| 7 | 8→10 | 96 (768) | — | Recovering; pivoting to AR=96 |

---

## 2. New Findings Since Previous Report

### 2.1 Value Embedding on All Layers (GPU 1, val_bpb: 0.9717 → 0.9711)

**Discovery:** GPU 1 changed `has_ve()` from an alternating pattern to `return True` (VE on every layer).

```python
# Before: alternating layers
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

# After: all layers
def has_ve(layer_idx, n_layer):
    return True
```

**Effect:** Increases parameter count from ~94M to ~119.5M (value_embeds: 50.3M params). Val_bpb improved by 0.0006 (0.97166 → 0.97109). The additional capacity from VE on all layers provides a consistent improvement at marginal VRAM cost (+1.7 GB).

**Mechanism:** Value Embeddings allow each token's value vector in attention to be modulated by the input token identity, effectively creating a form of key-value memory. Applying this to all layers rather than alternating ones doubles the expressiveness of this mechanism.

### 2.2 Value Embedding Weight Decay Tuning (GPU 1, val_bpb: 0.9711 → 0.9682)

**Discovery:** The VE parameters (50M parameters) were being regularized with the same weight_decay as other embeddings. Systematically reducing VE weight_decay revealed a monotonic improvement trend:

| VE weight_decay | val_bpb | Improvement |
|-----------------|---------|-------------|
| default (0.05) | 0.970797 | — |
| 0.03 | 0.969309 | −0.0015 |
| 0.02 | 0.968596 | −0.0007 |
| 0.01 | **0.968240** | −0.0004 |

The trend continued monotonically lower, suggesting VE parameters benefit from minimal regularization. This is architecturally motivated: value embeddings act as a lookup table (similar to token embeddings, which use weight_decay=0), so aggressive regularization harms their representational capacity.

**Implementation:**
```python
dict(kind='adamw', params=value_embeds_params,
     lr=embedding_lr * dmodel_lr_scale,
     betas=adam_betas, eps=1e-10,
     weight_decay=0.01),   # was 0.05; trend suggests 0.0 may be even better
```

### 2.3 DEPTH=7 Outperforms DEPTH=8 at AR=96 (GPU 4 & GPU 6)

**Discovery:** Both GPU 4 and GPU 6 independently found that reducing depth from 8 to 7 (at ASPECT_RATIO=96) improves val_bpb within the 5-minute budget.

| Config | DEPTH | Steps | val_bpb |
|--------|-------|-------|---------|
| AR=96, DEPTH=8 | 8 | ~1,200 | 0.9726 |
| AR=96, DEPTH=7 | 7 | ~1,312 | 0.9710 (GPU 4) / 0.9753 (GPU 6) |

**Mechanism:** With a fixed 5-minute wall-clock budget, a shallower model runs faster per step. DEPTH=7 with AR=96 achieves ~1,312 steps vs ~1,200 for DEPTH=8 — a **9.3% increase in optimizer steps** — more than compensating for the reduced model capacity.

This is a generalization of the batch size reduction finding (2^19→2^18 doubles steps): **within a fixed-time budget, step count dominates model capacity** up to a point.

GPU 4 further refined the DEPTH=7 configuration:
- Muon momentum ramp extended to 300 steps (better fit for 1,312-step runs): val_bpb 0.9710 → 0.9707
- softmax_scale=0.11 (explicit attention temperature): val_bpb fine-tuning ongoing

### 2.4 softmax_scale Tuning (GPU 4)

GPU 4 found that an explicit softmax scale of 0.125 (slightly above the default `1/sqrt(head_dim) ≈ 0.0884`) improves attention quality:

```python
# flash_attn_func call
y = fa3.flash_attn_func(q, k, v, softmax_scale=0.125, causal=True, window_size=window_size)
```

The optimal scale was found via systematic search: 0.09 < 0.10 < 0.11 ≈ 0.125. Combined with WEIGHT_DECAY=0.18, this configuration achieved 0.9719 (best of GPU 4's previous phase). The current DEPTH=7 exploration has superseded this but softmax_scale will need re-tuning.

---

## 3. Complete GPU 1 Learning Trajectory

GPU 1 has the richest exploration history (229 experiments, 34 keep). The full progression of its best val_bpb:

| Exp # | val_bpb | Key Change |
|-------|---------|------------|
| 1 | 0.9979 | Baseline |
| 9 | 0.9955 | AR=80 (wider model) |
| 85 | 0.9890 | batch=2^18 (HUGE — +1000 steps) |
| 103 | 0.9822 | AR=96 (768-dim) |
| 131 | 0.9793 | SSSM window pattern |
| 132 | 0.9787 | resid_lambdas=0.9 (LayerScale) |
| 133 | 0.9780 | WARMDOWN_RATIO=0.9 |
| 136 | 0.9768 | UNEMBEDDING_LR=0.008 |
| 138 | 0.9752 | FINAL_LR_FRAC=0.03 |
| 143 | 0.9729 | QK-norm before RoPE |
| 144 | 0.9727 | ve_gate_channels=128 |
| 148 | 0.9725 | ADAM_BETAS beta2=0.9 |
| 150 | 0.9723 | Muon beta2=0.90 |
| 158 | 0.9719 | Muon momentum ramp 0.85→0.93 |
| 173 | 0.9717 | MATRIX_LR=0.055 |
| 195 | 0.9711 | VE on all layers (119.5M params) |
| 218 | 0.9708 | VE weight_decay=0.05 |
| 220 | 0.9693 | VE weight_decay=0.03 |
| 221 | 0.9686 | VE weight_decay=0.02 |
| 222 | **0.9682** | VE weight_decay=0.01 |

The trajectory shows three phases of discovery:
1. **Macro architecture** (exp 1–103): batch size, model width — large discrete gains
2. **Training recipe** (exp 131–173): attention pattern, LR schedule, optimizer — incremental but consistent gains
3. **Fine-grained regularization** (exp 195–222): VE coverage and regularization — sustained monotonic improvement

---

## 4. Cross-GPU Validation Matrix (Updated)

| Technique | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7 |
|-----------|------|------|------|------|------|------|------|------|
| batch=2^18 | ★ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AR=96 (768-dim) | ★ | ✓ | — | — | ✓ | — | ✓ | ✓* |
| short_window //16 | ★ | ✓ | — | — | ✓ | — | ✓ | — |
| QK-norm before RoPE | ✓ | ★ | — | — | ✓ | — | — | — |
| UNEMBEDDING_LR=0.008 | ✓ | ★ | — | — | ✓ | — | — | ✓ |
| FINAL_LR_FRAC=0.03 | ★ | ✓ | ✓ | — | ✓ | — | ✓ | — |
| WARMDOWN≥0.7 | ★ | ✓ | ✓ | — | ✓ | — | ✓ | — |
| beta2=0.90 (Muon) | ✓ | ✓ | — | — | ★ | — | — | — |
| SSSM pattern | ★ | ✓ | — | — | ✓ | — | — | — |
| resid_lambdas=0.9 | ★ | ✓ | — | ✓ | — | ✓ | — | — |
| softmax_scale=0.125 | — | — | — | — | ★ | — | — | — |
| **DEPTH=7+AR=96** | — | — | — | — | ★ | — | **✓** | — |
| **VE on all layers** | — | ★ | — | — | — | — | — | — |
| **VE weight_decay=0.01** | — | ★ | — | — | — | — | — | — |
| x0_lambdas highway | — | — | — | ✓ | — | ★ | — | — |
| RoPE base=50k+ | — | ✓ | — | ★ | ✓ | ✓ | — | — |
| DEPTH=9/10 (narrow) | — | — | ★ | ✓ | ✓ | ✓ | — | — |

(★ = first discovery, ✓ = confirmed independently, * = in progress)

---

## 5. Best Configurations

### 5.1 GPU 1 Best (val_bpb = 0.968240, commit `3388c96`)

```python
# Architecture
DEPTH = 8
ASPECT_RATIO = 96             # model_dim=768, ~119.5M params (VE on all layers)
TOTAL_BATCH_SIZE = 2**18      # ~1,200 steps/5min

# Attention
WINDOW_PATTERN = "SSSM"       # S=128 (//16), M=1024 (//2)
# QK-norm BEFORE RoPE (q, k = norm(q), norm(k) at line 89)
# RoPE base = 50,000

# Optimizer
MATRIX_LR = 0.055             # Muon
SCALAR_LR = 0.3
EMBEDDING_LR = 0.4
UNEMBEDDING_LR = 0.008
WEIGHT_DECAY = 0.15
ADAM_BETAS = (0.8, 0.9)
WARMDOWN_RATIO = 0.9
FINAL_LR_FRAC = 0.03
# Muon momentum: 0.85 → 0.93 ramp over 200 steps, beta2=0.90

# Model init
resid_lambdas_init = 0.9      # LayerScale dampening
x0_lambdas_init = 0.1
ve_gate_channels = 128
has_ve = all layers            # was alternating
VE_weight_decay = 0.01        # was 0.05; key new finding
```

### 5.2 GPU 4 Best (val_bpb = 0.970708, commit `d844491`)

```python
DEPTH = 7                     # shallower → ~1,312 steps/5min
ASPECT_RATIO = 96             # model_dim=768
TOTAL_BATCH_SIZE = 2**18
WINDOW_PATTERN = "SSSM"
WEIGHT_DECAY = 0.18
ADAM_BETAS = (0.82, 0.9)
WARMDOWN_RATIO = 0.75
FINAL_LR_FRAC = 0.03
MATRIX_LR = 0.04
# softmax_scale=0.125 (in attention)
# Muon momentum ramp over 300 steps (fits 1312 steps better)
```

---

## 6. Discussion

### Why Does VE Weight Decay Matter?

Value Embeddings are a form of token-conditioned attention: for each input token, a small MLP generates an additive bias to the value vector. Functionally, they resemble word embeddings — they encode token identity into the attention mechanism.

The standard token embedding (`wte`) uses `weight_decay=0` precisely because excessive regularization prevents the model from specializing each token's representation. Applying the same logic to VE parameters: lower weight decay allows each token's VE to maintain a distinct signature. The monotonic improvement from 0.05 → 0.03 → 0.02 → 0.01 strongly suggests the optimal value is near zero, and `weight_decay=0.0` is the natural next test.

### DEPTH=7 vs DEPTH=8: Steps vs Capacity

The finding that DEPTH=7 outperforms DEPTH=8 at identical model_dim=768 is not a statement about the optimal architecture for language modeling — it is a statement about the **5-minute wall-clock constraint**. With more training compute (longer runs), DEPTH=8 would likely recover and surpass DEPTH=7.

This is the clearest demonstration of the "within-budget optimization" principle that has governed this entire search:

```
Effective improvement = (steps gained) × (per-step improvement) − (capacity lost)
```

For DEPTH=7 vs DEPTH=8 at AR=96:
- Steps gained: +112 steps (+9.3%)
- Capacity lost: ~17M parameters (−12%)
- Net effect: +0.0016 val_bpb improvement

The search has been, at its core, an optimization of this tradeoff at every level.

### Convergent Evidence for Core Findings

The following techniques have now been confirmed by ≥3 independent agents:

| Technique | Agents | Confidence |
|-----------|--------|------------|
| batch=2^18 | All 8 | Very High |
| AR=96 | GPU 0,1,4,6,7 | Very High |
| QK-norm before RoPE | GPU 0,1,4 | High |
| SSSM window | GPU 0,1,4 | High |
| UNEMBEDDING_LR=0.008 | GPU 0,1,4 | High |
| WARMDOWN≥0.7 | GPU 0,1,2,4,6 | Very High |
| DEPTH=7+AR=96 | GPU 4,6 | Medium (2 agents) |
| VE all layers | GPU 1 only | Low (1 agent) |
| VE WD=0.01 | GPU 1 only | Low (1 agent) |

---

## 7. Open Questions and Next Steps

The following high-value experiments have not yet been run or require re-validation on the current best configuration:

1. **VE weight_decay = 0.0** — trend suggests this is the natural next step for GPU 1
2. **DEPTH=7 + VE on all layers** — combine GPU4's depth finding with GPU1's VE finding
3. **softmax_scale on GPU 1** — GPU 4 found 0.11–0.125 optimal; GPU 1 has not tested this
4. **x0_lambdas > 0.1** — GPU 5's highway connection finding (0.6 is optimal on GPU5) has not been tested on the strong AR=96 base config
5. **DEPTH=6 at AR=96** — continuing the depth reduction trend
6. **AR=112 (model_dim=896)** — next width step, ~76GB VRAM (within H100 limit)

---

## 8. Conclusion

After ~26 hours and 1,444 total experiments across 8 H100 GPUs, the search has improved val_bpb from the baseline 0.9979 to **0.9682** — a **3.0% relative improvement**. The most recent gains (from 0.9717 to 0.9682) came from two focused discoveries by GPU 1: applying Value Embeddings to all transformer layers and aggressively reducing VE weight decay.

**Cumulative experiment count by discovery phase:**

| Cumulative Exps | Best val_bpb | Key Discovery |
|----------------|-------------|---------------|
| 0 | 0.9979 | Baseline |
| ~50 | 0.9900 | batch=2^18 |
| ~250 | 0.9820 | AR=96 |
| ~400 | 0.9793 | SSSM + WARMDOWN |
| ~600 | 0.9729 | QK-norm before RoPE |
| ~900 | 0.9717 | MATRIX_LR + optimizer tuning |
| ~1,100 | 0.9707 | DEPTH=7 (GPU 4) |
| ~1,200 | 0.9711 | VE on all layers |
| ~1,444 | **0.9682** | VE weight_decay=0.01 |

The search remains active. GPU 1 and GPU 4 are the leading lines, exploring VE regularization and depth-efficiency tradeoffs respectively. The merge agent (separate machine) is tasked with validating and combining findings across branches.

---

*Report generated: 2026-03-09 09:52 UTC*
*Data source: `/newcpfs/user/yuchen/agent/autoresearch-gpu{0-7}/results.tsv`*
*Previous report: `autoresearch_report_mar9.md` (best: 0.9723)*
