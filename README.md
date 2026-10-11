# Empirical Analysis of Scale, Tokenization, and Attention in Hindi and Nepali

An end-to-end study of **small language models for Indic languages**, investigating how corpus quality, tokenizer vocabulary, Transformer architecture, and fine-tuning affect language modeling and reasoning performance.

## Overview

This project implements and analyzes two independent, decoder-only GPT-style Transformer language models for Hindi (higher-resource) and Nepali (lower-resource). Both models are built and trained from scratch using PyTorch, without pretrained language models or pretrained tokenizers.

The project covers the complete ML pipeline, from corpus collection and preprocessing to tokenizer training, language-model pretraining, reasoning fine-tuning, and mechanistic attention analysis.

## Key Highlights

- Built data collection and preprocessing pipelines for Hindi and Nepali, processing **7.75M and 11.96M cleaned documents**, respectively.
- Trained separate SentencePiece UNIGRAM tokenizers with vocabulary sizes of 8,000 (Hindi) and 4,000 (Nepali).
- Implemented two GPT-style Transformer language models from scratch, each with approximately **25M trainable parameters**.
- Pretrained both models for 16,000 steps on a single NVIDIA T4 GPU per run.
- Fine-tuned the models on six synthetic comparative and reasoning task categories.
- Investigated attention-head specialization, tokenizer fertility, catastrophic forgetting, and the effect of removing Rotary Position Embeddings (RoPE).

## Model Architecture

Both models use a decoder-only Transformer architecture implemented using PyTorch.

| Component | Configuration |
|---|---|
| Architecture | GPT-style decoder-only Transformer |
| Transformer layers | 8 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Context length | 512 tokens |
| Feed-forward activation | SwiGLU |
| Positional encoding | Rotary Position Embeddings (RoPE) |
| Normalization | Pre-Layer Normalization |
| Optimizer | AdamW |
| Training steps | 16,000 per model |
| Precision | FP16 mixed precision |
| Hardware | NVIDIA Tesla T4, 16 GB VRAM |

The models have independent corpora, tokenizers, vocabularies, and weights. Their feed-forward dimensions were calibrated to account for differences in vocabulary size and maintain comparable parameter counts.

## Data Processing and Tokenization

The preprocessing pipeline includes:

1. Short-document filtering.
2. FastText-based language identification.
3. Unicode NFC normalization.
4. Devanagari text sanitization.
5. Exact deduplication using SHA-256 hashes.
6. Near-duplicate removal using MinHash LSH.
7. Deterministic train, validation, and test splitting.

### Tokenizer comparison

| Metric | Hindi | Nepali |
|---|---:|---:|
| Cleaned documents | 7.75M | 11.96M |
| Vocabulary size | 8,000 | 4,000 |
| Token fertility (tokens/word) | 1.380 | 1.698 |
| Unknown-token rate | 0.1508% | 0.1098% |

The experiments examine how vocabulary size and subword fragmentation affect language modeling and downstream reasoning, particularly for rare entities and proper nouns.

## Experimental Results

### 1. Language Modeling

Evaluation was conducted on held-out test data using loss, perplexity, and Bits-Per-Byte (BPB).

| Metric | Hindi | Nepali |
|---|---:|---:|
| Test loss | 3.2946 | 3.1785 |
| Test perplexity | 26.97 | 24.01 |
| Test BPB | 0.5101 | 0.5112 |
| Training time | 4.7 hours | 4.8 hours |

Despite different vocabulary sizes and resource availability, both models achieved nearly identical BPB scores, providing a more comparable measure of compression across the two tokenizations.

### 2. Comparative Reasoning Fine-Tuning

Both pretrained models were fine-tuned on six task categories using synthetic data generated in the respective languages.

Tasks include:

- Direct comparison
- Transitive ordering
- Multi-hop chained reasoning
- Negated comparisons
- Multi-attribute comparisons
- Superlative selection

Evaluation used exact-match accuracy on 1,500 held-out questions per model.

| Metric | Hindi | Nepali |
|---|---:|---:|
| Pretrained reasoning accuracy | 0.00% | 0.00% |
| Fine-tuned reasoning accuracy | 62.60% | 54.33% |
| Selected fine-tuning dataset size | 5K | 7K |
| Perplexity degradation after fine-tuning | 9.42% | 19.52% |

The experiments demonstrate the trade-off between acquiring task-specific reasoning capabilities and retaining pretrained language-model performance.

### 3. Attention Analysis and Ablation

Attention matrices were analyzed using attention entropy, mean attention distance, and heatmaps across Transformer layers and heads.

Key findings:

- Identified local, pointer-like, long-range, and BOS-anchoring attention patterns.
- Observed a maximum mean attention distance of 12.94 tokens in a Nepali attention head.
- Analyzed changes in attention behavior before and after reasoning fine-tuning.
- Conducted a No-RoPE ablation on the Hindi model to investigate the role of positional information.

**No-RoPE ablation:** Removing RoPE increased Hindi test perplexity from 26.97 to 30.45, a 12.91% degradation. Attention analysis also revealed that two middle-layer heads collapsed toward immediate-predecessor attention.

## Tech Stack

- **Programming:** Python
- **Deep Learning:** PyTorch
- **Tokenization:** SentencePiece UNIGRAM
- **Data Processing:** FastText, Unicode normalization, SHA-256, MinHash LSH
- **Training:** AdamW, learning-rate scheduling, FP16 mixed precision
- **Evaluation:** Perplexity, BPB, BLEU-4, chrF++, ROUGE-L, exact-match accuracy
- **Visualization:** Matplotlib, attention heatmaps, entropy and attention-distance analysis
- **Compute:** Kaggle NVIDIA Tesla T4 GPU

## Project Structure

The repository is organized around the main stages of the ML pipeline. Adapt the paths below to match the actual files in the repository.

```text
indic-transformer-lm-analysis/
├── hindi/
│   ├── data/
│   ├── tokenizer/
│   ├── model/
│   ├── train/
│   ├── finetune/
│   ├── evaluation/
│   └── configs/
├── nepali/
│   ├── data/
│   ├── tokenizer/
│   ├── model/
│   ├── train/
│   ├── finetune/
│   ├── evaluation/
│   └── configs/
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

Large datasets, tokenized corpora, and model checkpoints may be distributed separately, with links provided in the repository when available.

## Reproducibility

The project includes the implementation, training configurations, evaluation results, and analysis artifacts needed to document the experiments.

To reproduce the experiments:

1. Set up the Python environment and install the dependencies.
2. Prepare the Hindi and Nepali corpora using the preprocessing pipelines.
3. Train the language-specific SentencePiece tokenizers.
4. Initialize and pretrain the two Transformer models independently.
5. Generate synthetic reasoning datasets and fine-tune the pretrained checkpoints.
6. Run language-modeling evaluation, reasoning evaluation, and attention-analysis experiments.

See the implementation scripts and configuration files for the exact hyperparameters and execution commands.

## Research Questions

This project investigates:

- How does tokenizer vocabulary size affect subword fragmentation and model behavior?
- Can small Transformers achieve comparable language-modeling efficiency across different resource settings?
- How does reasoning fine-tuning affect task performance and language-model retention?
- What changes occur in attention-head behavior after fine-tuning?
- How does removing explicit positional information affect language modeling?

## Author

**Srinivas Billuri**  
B.Tech, Computer Science and Engineering  
International Institute of Information Technology, Hyderabad (IIITH)

---

*This project was completed as part of the Language Models and Agents individual project, Monsoon 2026.*
