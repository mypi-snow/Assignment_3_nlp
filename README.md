# Toxic Comment Detection using Bidirectional LSTM with Dual Embeddings

**NLP Assessment 3 — University of Technology Sydney (42XXX Natural Language Processing)**

| Member | Student ID | Contribution |
|--------|-----------|--------------|
| Pearl Jennifer | 25715386 | Problem definition, Introduction & Background, dataset acquisition, class imbalance analysis |
| Yash Mehta | 25731035 | Methodology, model architecture design, GloVe embedding pipeline, Bi-LSTM implementation |
| Raj Chhadia | 25215272 | Experiment Results, training loop, evaluation metrics, visualisation, report assembly |

**Submission Date:** Friday, 22 May 2026  
**Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1kXfZ5ek4e-7JU6XNM43S9l6JBgAZbi9J?authuser=1)

---

## Overview

This project addresses automated **toxic comment detection** on online platforms — a critical challenge for content moderation at scale. We propose a **Bidirectional LSTM (Bi-LSTM) with Dual Embeddings** architecture that combines:

- **Pretrained GloVe 50d embeddings** — for rich distributional semantic knowledge
- **Trainable 50d embeddings** — to specialise in surface patterns of online toxicity
- **Bidirectional LSTM encoder** — for full left-and-right context awareness per token

We compare the proposed model against a **Simple LSTM baseline** on a balanced subset of the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset.

---

## Results Summary

| Model | Accuracy | F1 (macro) | Precision | Recall |
|-------|----------|-----------|-----------|--------|
| Simple LSTM (Baseline) | 65.47% | 0.6465 | 0.67 | 0.65 |
| **Bi-LSTM + Dual Embed (Proposed)** | **88.20%** | **0.8819** | **0.88** | **0.88** |

The proposed model delivers a **+22.73 percentage point** improvement in accuracy and achieves balanced precision/recall across both classes, dramatically reducing both false positives and false negatives.

### Confusion Matrix Highlights

| | Baseline LSTM | Proposed Bi-LSTM |
|-|--------------|-----------------|
| True Non-toxic correctly classified | 377 / 750 (50.3%) | 686 / 750 (91.5%) |
| True Toxic correctly classified | 605 / 750 (80.7%) | 637 / 750 (84.9%) |
| False Positives | 373 | 64 |
| False Negatives | 145 | 113 |

---

## Repository Structure

```
nlp-toxic-detection/
├── NLP_A3_Toxic_Comment_Detection.ipynb   # Main Colab notebook (all code)
├── README.md                               # This file
```

---

## Getting Started

### Requirements

The notebook is self-contained and installs all dependencies within Google Colab. No local setup is required beyond a Kaggle API key.

**Key libraries used:**
- `torch` (PyTorch) — model definition and training
- `kagglehub` — dataset download
- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualisation
- `scikit-learn` — evaluation metrics

### Running the Notebook

1. Open the notebook in Google Colab: [link](https://colab.research.google.com/drive/1kXfZ5ek4e-7JU6XNM43S9l6JBgAZbi9J?authuser=1)
2. Add your Kaggle API credentials when prompted (username + API key from [kaggle.com/account](https://www.kaggle.com/account))
3. Run all cells in order (`Runtime > Run all`)
4. Total runtime: **~25–30 minutes** on Colab free tier (CPU)

> No GPU is required. The 10,000-sample balanced subset is designed to fit within free Colab resource limits.

---

## Model Architecture

### Proposed: Bi-LSTM with Dual Embeddings

```
Input tokens (seq_len=100)
       │
       ├──► GloVe Embedding (50d, fine-tunable)  ─┐
       │                                           ├─► Concatenate (100d) ─► Dropout(0.4)
       └──► Trainable Embedding (50d, Xavier)     ─┘
                                                           │
                                              Bidirectional LSTM
                                              (hidden=128, 1 layer)
                                             ┌─────┴─────┐
                                          Forward      Backward
                                         h_T (128d)   h_1 (128d)
                                             └─────┬─────┘
                                           Concat (256d) ─► Dropout(0.4)
                                                   │
                                         Linear(256 → 2) ─► Softmax
                                                   │
                                           [Non-toxic, Toxic]
```

**Trainable parameters:** 1,451,134

### Baseline: Simple LSTM

```
Input tokens ─► Trainable Embedding (50d) ─► LSTM (hidden=128) ─► Dropout ─► Linear(128 → 2)
```

**Trainable parameters:** 699,968

---

## Dataset

- **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle)
- **Full dataset:** 159,571 Wikipedia talk page comments with 6 toxicity labels
- **Balanced subset used:** 10,000 samples (5,000 toxic + 5,000 non-toxic)
- **Split:** 70% train (7,000) / 15% val (1,500) / 15% test (1,500), stratified

### Preprocessing Pipeline

1. **Lowercase** all text
2. **Remove URLs** (HTTP/HTTPS)
3. **Remove punctuation** (retain alphabetic tokens only)
4. **Tokenise** on whitespace
5. **Truncate/pad** to 100 tokens
6. **Build vocabulary** from training data only (min freq = 2), with `<PAD>` and `<UNK>` tokens
7. **GloVe coverage:** 93.0% of vocabulary (11,298 / 12,151 words)

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=2) |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| Epochs | 10 |
| Dropout | 0.4 |
| Sequence length | 100 |

---

## Qualitative Inference Examples (Proposed Model)

| Comment | Prediction | Confidence |
|---------|-----------|-----------|
| "You are an absolute idiot and should crawl back..." | Toxic | 99.93% |
| "I really enjoyed reading your article, thanks..." | Non-toxic | 99.88% |
| "Go kill yourself, nobody wants you here." | Toxic | 99.68% |
| "The weather today is really nice, perfect for a walk." | Non-toxic | 99.28% |
| "You're so stupid it's actually impressive." | Toxic | 99.75% |
| "Great point! I hadn't considered that perspective before." | Non-toxic | 99.80% |

---

## Key References

- Harris, Z. S. (1954). Distributional structure. *Word, 10*(2–3), 146–162.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Jigsaw/Conversation AI. (2018). *Toxic Comment Classification Challenge*. Kaggle.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv*.
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Trans. Signal Processing, 45*(11).
- Vogels, E. A. (2021). The state of online harassment. *Pew Research Center*.

---

## License

This project was developed for academic assessment purposes at the University of Technology Sydney. The Jigsaw dataset is subject to its own [Kaggle competition rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules). GloVe embeddings are distributed under the [PDDL license](https://nlp.stanford.edu/projects/glove/).
