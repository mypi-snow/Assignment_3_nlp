# Toxic Comment Detection Using a Bidirectional LSTM with Dual Embeddings

**NLP Assessment 3 — University of Technology Sydney**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kXfZ5ek4e-7JU6XNM43S9l6JBgAZbi9J?authuser=1)

| Member | Student ID | Contribution |
|--------|-----------|--------------|
| Pearl Jennifer | 25715386 | Problem definition, Introduction & Background, dataset acquisition, exploration, and class imbalance analysis |
| Yash Mehta | 25731035 | Methodology, model architecture design, GloVe embedding pipeline, Bi-LSTM implementation, and debugging |
| Raj Chhadia | 25215272 | Experiment Results, training loop, evaluation metrics, visualisation, report assembly, and GitHub setup |

---

## Overview

Online platforms rely on user participation, but toxic and abusive comments can make public discussion unsafe and unwelcoming. Manual moderation cannot realistically keep pace with the volume of comments on large platforms. This project develops a compact NLP system for **binary toxic comment detection** using the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

We propose a **Bidirectional LSTM (Bi-LSTM) with Dual Embeddings**, combining:
- A **pretrained GloVe 50d embedding stream** — for broad semantic knowledge from large-scale corpora
- A **trainable 50d embedding stream** — to specialise in the language patterns of online toxicity
- A **Bidirectional LSTM encoder** — to capture full left-and-right sentence context before classification

The model is evaluated against a simple unidirectional LSTM baseline on a balanced 10,000-comment subset, designed to run end-to-end on free Colab CPU resources.

---

## Results

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Simple LSTM (Baseline) | 65.47% | 0.6465 | 0.67 | 0.65 |
| **Bi-LSTM + Dual Embeddings (Proposed)** | **88.20%** | **0.8819** | **0.88** | **0.88** |

*Test set: n = 1,500, balanced 50/50.*

The proposed model delivers a **+22.73 percentage point** improvement in accuracy. It also converges faster — reaching 84.40% validation accuracy after just one epoch versus the baseline's slow climb to a peak of 68.40%.

### Confusion Matrix Breakdown

| | Baseline LSTM | Proposed Bi-LSTM |
|-|--------------|-----------------|
| Non-toxic correctly classified | 377 / 750 (50.3%) | 686 / 750 (91.5%) |
| Toxic correctly classified | 605 / 750 (80.7%) | 637 / 750 (84.9%) |
| False Positives | 373 | 64 |
| False Negatives | 145 | 113 |

The proposed model dramatically reduces false positives — important for moderation because flagging too many harmless comments creates unnecessary review work and frustrates users.

---

## Repository Structure

```
Assignment_3_nlp/
├── NLP_A3_Toxic_Comment_Detection.ipynb   # Main notebook (all code, end-to-end)
├── NLP_A3_Report_Rebuilt_APA.docx         # Project report
└── README.md                              # This file
```

---

## Getting Started

The notebook is self-contained and installs all dependencies within Google Colab. All you need is a **Kaggle API key**.

### Steps

1. Open the notebook via the Colab badge above
2. Add your Kaggle credentials when prompted (username + API key from [kaggle.com/account](https://www.kaggle.com/account))
3. Run all cells (`Runtime → Run all`)
4. Total runtime: **~25–30 minutes** on Colab free tier (CPU, no GPU required)

### Key Libraries

| Library | Purpose |
|---------|---------|
| `torch` (PyTorch) | Model definition, training loop |
| `kagglehub` | Dataset download via Kaggle API |
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `seaborn` | Training curves, confusion matrices |
| `scikit-learn` | Accuracy, F1, classification report |

---

## Model Architecture

### Proposed: Bi-LSTM with Dual Embeddings (~1.45M parameters)

```
Input tokens (seq_len = 100)
       │
       ├──► GloVe Embedding (50d, fine-tunable)  ─┐
       │                                           ├─► Concatenate (100d) ─► Dropout(0.4)
       └──► Trainable Embedding (50d, random)     ─┘
                                                           │
                                              Bidirectional LSTM
                                              (hidden = 128, 1 layer)
                                             ┌──────┴──────┐
                                          Forward       Backward
                                         h_T (128d)    h_1 (128d)
                                             └──────┬──────┘
                                           Concat (256d) ─► Dropout(0.4)
                                                   │
                                         Linear(256 → 2) ─► Softmax
                                                   │
                                           [Non-toxic, Toxic]
```

### Baseline: Simple LSTM (~0.70M parameters)

```
Input tokens ─► Trainable Embedding (50d) ─► Unidirectional LSTM (hidden=128) ─► Dropout ─► Linear(128 → 2)
```

The only differences between the two models are dual embeddings vs. single trainable embedding, and bidirectional vs. unidirectional LSTM — making the performance comparison clean and attributable.

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience = 2) |
| Gradient clipping | max norm 1.0 |
| Batch size | 64 |
| Epochs | 10 |
| Dropout | 0.4 |
| Sequence length | 100 tokens |
| Loss function | Cross-entropy |

---

## Dataset

- **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Jigsaw/Conversation AI, 2018)
- **Full dataset:** 159,571 Wikipedia talk page comments with 6 toxicity labels
- **Balanced subset used:** 10,000 samples (5,000 toxic + 5,000 non-toxic)
- **Split:** 70% train (7,000) / 15% val (1,500) / 15% test (1,500), stratified

### Preprocessing Pipeline

1. Lowercase all text
2. Remove URLs
3. Remove non-alphabetic characters; keep whitespace-separated tokens
4. Tokenise on whitespace
5. Truncate or pad to 100 tokens
6. Build vocabulary from training split only (min frequency = 2), with `<PAD>` and `<UNK>` tokens
7. GloVe coverage: **93.0%** of vocabulary (11,298 / 12,151 words)

---

## Qualitative Inference Examples

| Comment | Prediction | Confidence |
|---------|-----------|-----------|
| "You are an absolute idiot and should crawl back under your rock." | Toxic | 99.93% |
| "I really enjoyed reading your article, thanks for sharing it." | Non-toxic | 99.88% |
| "Go kill yourself, nobody wants you here." | Toxic | 99.68% |
| "The weather today is really nice, perfect for a walk." | Non-toxic | 99.28% |
| "You're so stupid it's actually impressive." | Toxic | 99.75% |
| "Great point! I hadn't considered that perspective before." | Non-toxic | 99.80% |

---

## Limitations & Future Work

- The balanced 50/50 subset does not reflect the real-world class distribution (~9.6% toxic), so deployment performance would need further testing with threshold tuning or class weighting
- Word-level tokenisation with non-alphabetic removal limits handling of obfuscated spelling and punctuation-based toxicity
- Fixed 100-token sequence length means very long comments lose information through truncation
- Future directions: multi-label classification across all 6 Jigsaw categories; subword tokenisation; transformer models such as BERT (Devlin et al., 2019)

---

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186. https://doi.org/10.18653/v1/N19-1423
- Harris, Z. S. (1954). Distributional structure. *Word, 10*(2–3), 146–162. https://doi.org/10.1080/00437956.1954.11659520
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Jigsaw/Conversation AI. (2018). *Toxic Comment Classification Challenge*. Kaggle. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*. https://arxiv.org/abs/1412.6980
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*, 1532–1543. https://aclanthology.org/D14-1162/
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*(11), 2673–2681. https://doi.org/10.1109/78.650093
- Vogels, E. A. (2021). *The state of online harassment*. Pew Research Center. https://www.pewresearch.org/internet/2021/01/13/the-state-of-online-harassment/
- Zhang, Z., Robinson, D., & Tepper, J. (2018). Detecting hate speech on Twitter using a convolution-GRU based deep neural network. *ESWC 2018*, 745–760. https://doi.org/10.1007/978-3-319-93417-4_48

---

## Generative AI Acknowledgement

Generative AI tools were used to assist with language refinement, structure checking, and editing suggestions for the written report. The group reviewed and revised all final content, verified code outputs against the notebook, and remains responsible for the submitted work.
