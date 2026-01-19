
# Multi-Survey Seq Model (GRU + Time/Channel Attention) with Baseline (y0) Constraints  
**Predict Δ outcomes for PHQ-9 / P4 / Loneliness from weekly check sequences, and analyze interactions + service patterns around high-attention weeks.**

---

## 1) Overview

This project trains an interpretable sequential model to predict **3-class delta outcomes** for multiple surveys:

\[
y_0(\text{PHQ-9}, \text{P4}, \text{Loneliness}) \;\rightarrow\; C_{1:T}(\text{weekly checks}) \;\rightarrow\; \Delta y \in \{\text{improved}, \text{same}, \text{worse}\}
\]

Key design points:

- **Cohort filtering:** only patients with **no missing weekly checks** and **all three surveys** (PHQ-9, P4, Loneliness) available for baseline + final.
- **Inputs:** weekly ordinal checks `check1..check6` (z-scored globally).
- **Services `service1..service16`:** **NOT used for model training**, but kept for **post-hoc analysis** (service patterns at high-attention weeks).
- **Model:** GRUCell +  
  - **Dynamic channel attention** (per week, which check channels matter)  
  - **Additive time attention** (which weeks matter)
- **Hard constraints:** disallow impossible Δ classes given baseline `y0`  
  - if `y0 == 0`, **improved** is impossible  
  - if `y0 == y_max`, **worse** is impossible  
- **Multi-task:** predicts ΔPHQ, ΔP4, ΔLon simultaneously (weighted loss).

---

## 2) Prediction Targets

Each survey is converted into a **3-class delta label** using baseline and final scores:

- `0 = improved` if final < baseline  
- `1 = same` if final == baseline  
- `2 = worse` if final > baseline  

Implementation: `delta3(y0, yT)`.

---

## 3) Data Requirements

### Inputs
- `raw_data1.csv` (weekly sequence)
  - Patient ID column: `menti_seq`
  - Time column: `reg_date`
  - Weekly checks: either `check*_value` columns or `check*` (excluding `_type`)
  - Services: `service1..service16` columns

- `raw_data2.csv` (survey results)
  - Patient ID column: `menti_seq`
  - `srvy_name` in {`PHQ-9`, `P4`, `Loneliness`}
  - `srvy_result` numeric
  - `reg_date` datetime

### Cohort filtering rules
A patient is kept only if:
1. Has at least `T` weeks (default `T=40`) and uses the **first T** weeks.
2. Weekly checks have **no missing values** across those T rows.
3. Each survey (PHQ-9, P4, Loneliness) has **>=2 records** to compute baseline and final.

---

## 4) Model Architecture

### Baseline embedding (y0)
- Separate embeddings for each survey baseline:
  - PHQ-9: 0..3 (+ unknown)
  - P4: 0..2 (+ unknown)
  - Loneliness: 0..1 (+ unknown)
- Concatenate to `y0e`, then map to initial hidden state `h0`.

### Dynamic channel attention (per time step)
For week `t`, compute attention weights across check channels:
\[
\alpha_t = \text{softmax}(f([c_t, y0e, h_{t-1}]))
\]
Then build a weighted check vector:
\[
\tilde{c}_t = c_t \odot \alpha_t
\]

### GRU update (per time step)
GRUCell input:
- raw checks `c_t`
- weighted checks `\tilde{c}_t`
- time embedding `pos_t`
- optional baseline embedding `y0e`

### Additive time attention (across weeks)
Produces a context vector:
\[
\text{context} = \sum_t a_t h_t
\]
Then a shared encoder feeds three heads:
- `head_phq`, `head_p4`, `head_lon` → logits of shape `(B,3)` each.

### Hard constraint masking (y0 → feasible Δ)
Before loss/eval:
- If `y0==0`, mask out class `improved`
- If `y0==y_max`, mask out class `worse`

Masked logits are filled with `-1e9` so probability mass cannot go there.

---

## 5) Training

### Loss
Multi-task masked cross-entropy:
L = (w_phq * CE_phq + w_p4 * CE_p4 + w_lon * CE_lon) / (w_phq + w_p4 + w_lon)

Extras:
- Optional **inverse-frequency class weights** for PHQ (smoothed by exponent `alpha=0.5`).

### Split
Train/val split is stratified by `dY_phq` (ΔPHQ).

### Optimizer + regularization
- AdamW, `lr=1e-3`, `weight_decay=1e-4`
- gradient clipping at 1.0
- early stopping on validation loss

---

## 6) Outputs (Artifacts)

All outputs are written to `OUT = figs_phq_multi_attn/` by default.

### Training & evaluation
- `best.pt` : best model weights (early stopping)
- `learning_multi.png` : train/val loss curve
- Confusion matrices:
  - `cm_PHQ9_multi.png`
  - `cm_P4_multi.png`
  - `cm_Loneliness_multi.png`
- Attention plots:
  - `attn_mean_val.png` : average time attention over val
  - `attn_by_dPHQ_val.png` : time attention by ΔPHQ group
  - `attn_by_y0PHQ_val.png` : time attention by baseline PHQ group
- Channel attention plots:
  - `ch_attn_overall_val.png`
  - `ch_attn_by_y0PHQ_val.png`
  - `ch_attn_by_dPHQ_val.png`

### Multi-survey interaction analysis
- `xtab_true_PHQ_P4.csv`, `xtab_true_PHQ_Loneliness.csv`
- `xtab_pred_PHQ_P4.csv`, `xtab_pred_PHQ_Loneliness.csv`
- `interactions_delta.json`:
  - true/pred crosstabs
  - conditional distributions
  - probability correlations (improved/worse probabilities across surveys)
- Baseline comorbidity:
  - `xtab_baseline_PHQ_P4.csv`
  - `xtab_baseline_PHQ_Loneliness.csv`

### Service analysis (post-hoc, not used in training)
- Basic usage stats:
  - `service_usage_overall_train.csv`
  - `service_usage_by_y0PHQ_train.csv`
  - `service_usage_by_dPHQ_train.csv`
- Attention-linked service patterns:
  - `service_top{K}_per_patient_val.csv`
  - `service_top{K}_by_dPHQ_val.csv`
  - `service_top{K}_by_improved_vs_non_val.csv`
  - `service_diff_improved_vs_non_top{K}_val.png`

### Service → outcome association (logistic regression)
- Top-K attention window service features:
  - `service_top{K}_features_val.csv`
- Adjusted OR table (requires `statsmodels`):
  - `service_logit_OR_top{K}_val.csv`
- Patient-level explanation table:
  - `service_patient_explanations_top{K}_val.csv`

### Service pairs / sequences around high-attention weeks
- Sequence + pair features:
  - `service_seq_features_val.csv`
- Univariate OR for features:
  - `service_OR_seq_val.csv`
  - `service_OR_seq_top20_val.png`
- High-attention vs anytime comparison:
  - `service_seq_features_highattn_val.csv`
  - `service_OR_seq_highattn_val.csv`
- Pair/sequence OR summary:
  - `service_pair_OR_summary_val.csv`

---

## 7) Installation

### Recommended environment
- Python 3.10+
- PyTorch (CUDA optional)
- NumPy, Pandas, scikit-learn, matplotlib
- Optional: statsmodels (for adjusted odds ratios)

```bash
pip install numpy pandas scikit-learn matplotlib torch
pip install statsmodels   # optional, for OR analysis
````

---

## 8) Project Structure

```
.
├── data/
│   ├── raw_data1.csv
│   └── raw_data2.csv
├── figs_phq_multi_attn/           # generated outputs
└── phq_multi_seq.py               # main script (this code)
```

---

## 9) How to Run

1. Put your data files under `data/`:

* `data/raw_data1.csv`
* `data/raw_data2.csv`

2. Run:

```bash
python phq_multi_seq.py
```

Key knobs (edit in `__main__`):

* `T = 40` : sequence length
* `TRAIN = True/False` : train new model vs load checkpoint
* `epochs` (default uses 300 in main)
* `attn_quantile` for high-attention window features (default 0.8)

---

## 10) Interpretation Guide

### Time attention

* `attn[t]` shows which weeks contributed most to the endpoint prediction.
* Group plots (`attn_by_dPHQ`, `attn_by_y0PHQ`) reveal whether improvement/worsening is associated with early vs late weeks.

### Channel attention

* `ch_alpha[t, k]` shows which **check channels** were most relevant at week `t`.
* Group plots show whether baseline severity or outcome group shifts attention to specific checks.

### Services

Services are not used in training; they are analyzed **after** the model learns which weeks are important.
This allows questions like:

* “Which services tend to occur around high-attention weeks for improved vs non-improved patients?”
* “Do certain service pairs co-occur specifically during high-attention windows?”

---

## 11) Notes / Limitations

* This pipeline uses only **first and last** survey records to define Δ. Intermediate survey trajectories are not modeled.
* The hard y0 constraint improves plausibility but may also hide model miscalibration if baseline coding is noisy.
* Service analyses are associative (logistic OR) and not causal.

---

## 12) Citation / Acknowledgement

If you use this code for research, please cite it as an internal implementation of:

* GRU-based sequence modeling
* additive attention for time interpretability
* dynamic channel attention for feature-level interpretability
* baseline-dependent hard constraints for feasible outcome transitions

```

