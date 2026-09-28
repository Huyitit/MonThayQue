Below is a complete LaTeX report outline for your RNN sentiment analysis project on the IMDB dataset. It is designed for a beginner but follows a scientific report structure: theory first, then implementation, experiments, and training-progression evidence (diagrams, statistics, tables).

I also include a minimal LaTeX skeleton and tips for exporting plots from Jupyter.

---

## 📄 Suggested LaTeX Report Outline

### 1. Title Page
- **Title:** Sentiment Analysis on IMDB Reviews using a Recurrent Neural Network
- **Author:** Your Name
- **Date:** \today
- **Abstract:** 150–250 words summarising dataset, model, key results.

### 2. Introduction
- **2.1 Motivation:** Why sentiment analysis? Why RNNs?
- **2.2 Problem Statement:** Binary classification of movie reviews (positive/negative).
- **2.3 Objectives:** Train a SimpleRNN/LSTM from scratch, evaluate performance, analyse training progression.
- **2.4 Contributions:** Implementation, training curves, error analysis.
- **2.5 Report Structure:** Brief roadmap.

### 3. Theoretical Background
- **3.1 Sentiment Analysis:** Definition, applications.
- **3.2 Neural Network Basics:** Layers, activation functions, loss.
- **3.3 Recurrent Neural Networks (RNNs)**
  - **Figure:** Unfolded RNN diagram.
  - **Equations:**
    ```latex
    \begin{equation}
    h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
    \end{equation}
    \begin{equation}
    \hat{y} = \sigma(W_{hy} h_T + b_y)
    \end{equation}
    ```
- **3.4 Backpropagation Through Time (BPTT)**
  - Vanishing/exploding gradient problem.
- **3.5 LSTM and GRU (Optional but Recommended)** - Do not include in my report
  - **Figure:** LSTM cell diagram.
  - **Equations:** Forget gate, input gate, output gate, cell state.
- **3.6 Word Embeddings:** Dense vector representations.
- **3.7 Binary Cross-Entropy Loss:**
    ```latex
    \begin{equation}
    \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)
    \end{equation}
    ```

### 4. Dataset and Preprocessing
- **4.1 Dataset Description:** IMDB 50K movie reviews.
  - **Table:** Number of reviews, train/test split, class balance.
- **4.2 Exploratory Data Analysis (EDA)**
  - **Figure:** Class distribution (bar chart).
  - **Figure:** Review length histogram.
  - **Table:** Mean, median, max review length.
- **4.3 Text Preprocessing**
  - Tokenisation, vocabulary building, padding.
  - **Table:** Vocabulary size, max sequence length, OOV rate.
  - **Equations:** Integer encoding, padding.

### 5. Model Architecture and Implementation
- **5.1 Overview**
  - **Figure:** Model architecture diagram (Embedding → RNN → Dense).
- **5.2 Embedding Layer:** `input_dim`, `output_dim`, `mask_zero`.
- **5.3 RNN Layer:** `SimpleRNN` or `LSTM`, units.
- **5.4 Dense Output Layer:** 1 unit, sigmoid activation.
- **5.5 Implementation Details**
  - Framework: TensorFlow/Keras.
  - **Table:** Hyperparameters (embedding dim, hidden units, batch size, learning rate, epochs).
  - **Table:** Trainable parameter counts per layer.

### 6. Experimental Setup
- **6.1 Training Configuration:** Optimizer (Adam), loss, metrics.
- **6.2 Hardware/Software:** CPU/GPU, Python, TensorFlow version.
- **6.3 Evaluation Metrics:** Accuracy, Precision, Recall, F1, AUC.

### 7. Results and Training Progression
- **7.1 Training Curves**
  - **Figure:** Training vs validation loss per epoch.
  - **Figure:** Training vs validation accuracy per epoch.
  - **Table:** Epoch-wise metrics (loss, accuracy, val_loss, val_accuracy).
- **7.2 Final Evaluation on Test Set**
  - **Table:** Test accuracy, precision, recall, F1, AUC.
  - **Figure:** Confusion matrix.
  - **Figure:** ROC curve.
- **7.3 Sample Predictions**
  - **Table:** Review snippet, true label, predicted probability.
- **7.4 Error Analysis**
  - Misclassified examples, discussion of failure modes.

### 8. Discussion
- Interpretation of training curves (overfitting/underfitting).
- Effect of hyperparameters (vocab size, embedding dim, RNN units).
- Limitations of SimpleRNN vs LSTM.
- Comparison with baseline (e.g., logistic regression on TF-IDF).

### 9. Conclusion and Future Work
- Summary of findings.
- Future improvements: LSTM/GRU, pretrained embeddings (GloVe), hyperparameter tuning, attention.

### 10. References
- Dataset: Kaggle IMDB.
- TensorFlow/Keras documentation.
- Papers: LSTM (Hochreiter & Schmidhuber), RNN tutorials.

### 11. Appendix
- Code snippets (using `listings`).
- Additional plots.
- Full hyperparameter table.

---

## 🧩 Minimal LaTeX Skeleton

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{subcaption}
\usepackage{listings}
\usepackage{xcolor}

\title{Sentiment Analysis on IMDB Reviews using a Recurrent Neural Network}
\author{Your Name}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report presents the theory and implementation of a recurrent neural network (RNN) for binary sentiment classification on the IMDB dataset. We describe preprocessing, model architecture, training progression, and evaluation. The model achieves ... 
\end{abstract}

\section{Introduction}
...

\section{Theoretical Background}
...
\begin{equation}
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
\end{equation}

\section{Dataset and Preprocessing}
...
\begin{table}[h]
\centering
\begin{tabular}{lrr}
\toprule
Statistic & Train & Test \\
\midrule
Number of reviews & 25000 & 25000 \\
Positive reviews & 12500 & 12500 \\
Negative reviews & 12500 & 12500 \\
\bottomrule
\end{tabular}
\caption{Dataset statistics.}
\label{tab:dataset}
\end{table}

\section{Model Architecture}
...
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{rnn_unfolded.pdf}
\caption{Unfolded RNN architecture.}
\label{fig:rnn}
\end{figure}

\section{Results and Training Progression}
...
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{training_loss.pdf}
\caption{Training and validation loss per epoch.}
\label{fig:loss}
\end{figure}

\section{Discussion}
...
\section{Conclusion}
...
\bibliographystyle{plain}
\bibliography{references}

\appendix
\section{Code Snippets}
...

\end{document}
```

---

## 📊 Figures and Tables to Include

| Figure/Table | Content | Purpose |
|--------------|---------|---------|
| Fig. 1 | Unfolded RNN | Theory |
| Fig. 2 | LSTM cell | Theory |
| Fig. 3 | Class distribution | EDA |
| Fig. 4 | Review length histogram | EDA |
| Fig. 5 | Model architecture | Implementation |
| Fig. 6 | Training/validation loss | Training progression |
| Fig. 7 | Training/validation accuracy | Training progression |
| Fig. 8 | Confusion matrix | Final evaluation |
| Fig. 9 | ROC curve | Final evaluation |
| Table 1 | Dataset statistics | EDA |
| Table 2 | Hyperparameters | Implementation |
| Table 3 | Parameter counts | Implementation |
| Table 4 | Epoch-wise metrics | Training progression |
| Table 5 | Test set metrics | Final evaluation |
| Table 6 | Sample predictions | Error analysis |

---

## 💾 Exporting Plots from Jupyter for LaTeX

Use PDF for vector graphics (best for LaTeX):
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('assets/images/training_loss.pdf', bbox_inches='tight') # save outputs in assets folder and use it in latex file
plt.show()
```
Then in LaTeX: `\includegraphics[width=0.8\textwidth]{assets/images/training_loss.pdf}`.

For tables, you can generate LaTeX code from pandas:
```python
print(df.to_latex(index=False, float_format="%.4f"))
```

---

This outline gives you a clear, academic structure that highlights both the **theory** behind RNNs and the **empirical evidence** of training progression. You can adapt the depth of theory (e.g., include LSTM equations or keep SimpleRNN only) based on your audience.