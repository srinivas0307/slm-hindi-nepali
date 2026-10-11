# Hindi Reasoning Finetuning & Bonus Ablation Artifacts

## Google Drive Checkpoints & Model Weights
- **Google Drive Link:** [Hindi Finetuned Models & Checkpoints](https://drive.google.com/drive/folders/1Ck7WfBfOvz4q6Z4FTud0sJfZAxDE1SVA?usp=sharing)

This folder contains the saved `.pt` model weights for:
1. Sizing sweep finetuned models (`train_5k`, `train_6k`, `train_7k`, `train_8k`)
2. Best selected finetuned model checkpoint
3. No-RoPE ablated Hindi model checkpoint (`best_model.pt` from step 16,000)

---

## Directory Contents

| File / Directory | Description |
|---|---|
| `hindi-finetune.ipynb` | Kaggle notebook for multi-slice finetuning with dual-metric tracking (Reasoning accuracy + LM PPL). |
| `hindi-reasoing-eval.ipynb` | Test evaluation notebook on 1,500 held-out Devanagari reasoning examples (exact match accuracy). |
| `hindi-attention-eval.ipynb` | Layer 0, 4, 7 attention extraction toolkit generating entropy, distance, and delta heatmaps. |
| `hindi_samples_vs_ppl_tradeoff.png` | Pareto trade-off curve showing reasoning accuracy vs. language modeling perplexity across sample sizes. |
| `hindi_no_rope.ipynb` | Training notebook for pretraining Model H from scratch without Rotary Position Embeddings. |
| `no-rope-eval.ipynb` | Full Phase 2 benchmark evaluation notebook on the ablated No-RoPE model checkpoint. |
| `eval_hindi_no_rope/` | Evaluation outputs including `eval_results.json` and 14 attention visualization heatmaps. |

---

## Key Results Summary

- **Pretrained Reasoning Accuracy:** 0.00%
- **Finetuned Reasoning Accuracy (6k slice):** 62.60% (Transitive ordering: 96.88%, Multihop chained: 95.74%)
- **Optimal Training Sample Size:** 6,000 examples (maintains LM PPL at 29.71, +11.5% degradation vs. 26.97 baseline)
- **No-RoPE Ablation Degradation:** Test PPL degraded by +12.91% (26.97 $\to$ 30.45); Layer 4 Heads 6 & 7 collapsed into local shift-register diagonal attention bands (mean distance $\approx 1.16$).