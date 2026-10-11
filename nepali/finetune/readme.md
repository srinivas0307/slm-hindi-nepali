# Nepali Reasoning Finetuning Artifacts

## Google Drive Checkpoints & Model Weights
- **Google Drive Link:** [Nepali Finetuned Models & Checkpoints](https://drive.google.com/drive/folders/1OEUGUsxwgkoo7cVm-BMsv0CuxqNB0zTk?usp=sharing)

This folder contains the saved `.pt` model weights for:
1. Sizing sweep finetuned models (`train_5k`, `train_6k`, `train_7k`, `train_8k`)
2. Best selected finetuned model checkpoint (7k slice)

---

## Directory Contents

| File / Directory | Description |
|---|---|
| `finetune-nepali-path.ipynb` | Kaggle notebook executing the 5k, 6k, 7k, 8k sample sizing sweep with dual-metric tracking. |
| `finetune_nepali.py` | Standalone Python training script with AdamW cosine decay and validation evaluation. |
| `nepali-reasoning-eval.ipynb` | Test evaluation notebook on 1,500 held-out Devanagari reasoning examples (exact match accuracy). |
| `nepali-attention-eval.ipynb` | Layer 0, 4, 7 attention extraction toolkit generating entropy, distance, and delta heatmaps. |
| `nepali_samples_vs_ppl_tradeoff (1).png` | Pareto trade-off curve showing reasoning accuracy vs. language modeling perplexity across sample sizes. |

---

## Key Results Summary

- **Pretrained Reasoning Accuracy:** 0.00%
- **Finetuned Reasoning Accuracy (7k slice):** 54.33% (Multihop chained: 85.60%, Transitive ordering: 78.14%, Negated: 77.59%)
- **Optimal Training Sample Size:** 7,000 examples (peak accuracy of 52.00% on validation set, +19.5% LM PPL degradation, within 20% budget)
- **Attention Dynamics:** Layer 7 shows uniform expansion of attention distance across all 8 heads (+0.44 to +2.53 tokens) as late layers route context back to entity premises.