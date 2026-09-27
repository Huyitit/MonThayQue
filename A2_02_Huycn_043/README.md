# Assignment 02: From Data Representation to Deployable Intelligent Systems

This repository contains the complete pipeline and artifacts for **Assignment 02**, implementing three distinct intelligent machine learning applications from raw data representation to deployable web services.

---

## 1. Project Applications

1. **Diabetes Prediction (Classification)**
   - Location: `diabetes/`
   - Problem: Healthcare screening to predict diabetes onset from clinical indicators.
   - Input Representation: Clinical tabular feature matrix $X \in \mathbb{R}^{B \times d}$.

2. **House Price Prediction (Regression)**
   - Location: `house_price/`
   - Problem: Residential real estate valuation from structural, demographic, and location characteristics.
   - Input Representation: Encoded and normalized tabular feature matrix $X \in \mathbb{R}^{B \times d}$.

3. **E-Commerce Customer Behavior & Interest Discovery (Classification/Prediction)**
   - Location: `customer_behavior/`
   - Problem: Discovery of customer interests and purchasing behavior using tabular features and textual customer feedback/reviews.
   - Input Representation: Concatenated tabular features and text token/embedding vectors $X \in \mathbb{R}^{B \times (d_{\text{tab}} + d_{\text{text}})}$.

---

## 2. Repository Structure

```text
Assignment_02/
├── diabetes/
│   ├── data/                 # Raw and cleaned dataset files
│   ├── notebook/             # Jupyter experimentation notebook
│   ├── model/                # Serialized model and preprocessing pipeline (joblib)
│   ├── api/                  # FastAPI / Flask backend service
│   ├── web/                  # Web frontend interface
│   └── requirements.txt      # Dependency specification
├── house_price/
│   ├── data/
│   ├── notebook/
│   ├── model/
│   ├── api/
│   ├── web/
│   └── requirements.txt
├── customer_behavior/
│   ├── data/
│   ├── notebook/
│   ├── model/
│   ├── api/
│   ├── web/
│   └── requirements.txt
├── report/
│   └── Assignment_02.pdf     # Final technical report (approx. 10 pages)
├── problem_definitions.md    # Formal problem, objective, and dataset definitions
├── requirements.md           # Reproducibility guidelines & environment specifications
├── requirements.txt          # Global project dependencies
└── README.md
```

---

## 3. Getting Started & Reproducibility

Detailed instructions for setting up the environment, installing dependencies, reproducing the experiments, and running the deployment services are available in [`requirements.md`](./requirements.md).

For formal problem statements, feature lists, and dataset configurations, see [`problem_definitions.md`](./problem_definitions.md).
