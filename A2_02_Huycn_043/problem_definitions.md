# Problem Definitions & Objectives (Assignment 02)

This document contains the problem definitions, objectives, and dataset specifications for the three intelligent applications as required by **Part I** and sections **8.1, 9.1, and 10.1** of Assignment 02.

---

## 1. Summary of Applications (Part I Compact Table)

| Application | Dataset Name | Kaggle URL | Rows (N) | Features (d) | Target Variable | Problem Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes Prediction** | *[e.g., Pima Indians Diabetes]* | *[Insert URL]* | *[N]* | *[d]* | `Outcome` / Class | Binary Classification |
| **House Price Prediction** | Vietnam Housing Prices (`vn_house.dataset.csv`) | [Kaggle](https://www.kaggle.com/) | 30,229 | 11 | `Price` (Billion VND) | Continuous Regression |
| **E-Commerce Customer Behavior** | *[e.g., E-Commerce Customer Behavior / Reviews]* | *[Insert URL]* | *[N]* | *[d]* | *[e.g., Product Category / Churn / Segment]* | Classification / Prediction |

---

## 2. Application 1 — Diabetes Prediction

### 2.1 Problem Description
- **Real-world Healthcare Problem:**
  > The objective of this application is to predict whether a patient is likely to have diabetes based on clinical and demographic indicators (e.g., glucose level, blood pressure, insulin, BMI, age). The prediction target is binary diabetes diagnosis ($y \in \{0, 1\}$). The prediction can potentially support early clinical screening and proactive lifestyle intervention.
- **Formal Definition:**
  - $X \in \mathbb{R}^{N \times d}$: Matrix of patient clinical attributes
  - $y \in \{0, 1\}^N$: Ground truth diabetes diagnostic outcome ($0 = \text{Non-diabetic}, 1 = \text{Diabetic}$)
  - Objective: Learn a hypothesis function $f_\theta: \mathbb{R}^d \to [0, 1]$ estimating $P(y = 1 \mid x)$.

### 2.2 Kaggle Dataset Details
- **Dataset Name:** *[Insert Name]*
- **Kaggle URL:** *[Insert URL]*
- **Observations ($N$):** *[e.g., 768]*
- **Attributes / Features ($d$):** *[e.g., 8]*
- **Target Variable:** `Outcome` (0 or 1)
- **Description of one observation:** A single patient record containing diagnostic physiological measurements and diagnosis status.

### 2.3 Feature Specification Table
| Feature | Type (Numerical / Categorical) | Description & Clinical Meaning |
| :--- | :--- | :--- |
| `Pregnancies` | Numerical (Discrete) | Number of times pregnant |
| `Glucose` | Numerical (Continuous) | Plasma glucose concentration after 2h oral glucose tolerance test |
| `BloodPressure`| Numerical (Continuous) | Diastolic blood pressure (mm Hg) |
| `SkinThickness`| Numerical (Continuous) | Triceps skin fold thickness (mm) |
| `Insulin` | Numerical (Continuous) | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Numerical (Continuous) | Body mass index (weight in kg / (height in m)²) |
| `DiabetesPedigreeFunction` | Numerical (Continuous) | Diabetes pedigree function (genetic score) |
| `Age` | Numerical (Discrete) | Age in years |
| *[Add/adjust]* | *[...] * | *[...] * |

---

## 3. Application 2 — House Price Prediction

### 3.1 Problem Description
- **Real-world Problem:**
  > The objective of this application is to predict the market selling price of a residential property based on its physical and spatial attributes (living area, number of bedrooms/bathrooms, location, property condition). The prediction target is a continuous real-valued price ($y \in \mathbb{R}^+$). This supports real estate buyers, sellers, and financial institutions in estimating fair property valuation.
- **Why it differs from Diabetes Classification:**
  - The target is continuous ($\mathbb{R}$) rather than discrete categories ($\{0, 1\}$).
  - Evaluated using error magnitudes (MAE, RMSE, $R^2$) rather than classification metrics (Accuracy, F1, AUC).
- **Formal Definition:**
  - $X \in \mathbb{R}^{N \times d}$: Computational feature matrix constructed from numerical and encoded categorical characteristics.
  - $y \in \mathbb{R}^N$: Continuous house price vector.
  - Objective: Learn $f_\theta: \mathbb{R}^d \to \mathbb{R}$ minimizing prediction error loss (e.g. MSE / Huber loss).

### 3.2 Kaggle Dataset Details
- **Dataset Name:** Vietnam Housing Prices Dataset (`vn_house.dataset.csv`)
- **Kaggle URL:** https://www.kaggle.com/datasets (Vietnam Real Estate Listings)
- **Observations ($N$):** 30,229 property listings
- **Attributes / Features ($d$):** 11 predictors + 1 continuous target (`Price`)
- **Target Variable:** `Price` (Continuous, Billion VND, range $[1.0, 11.5]$, $\mu = 5.87$, median $= 5.90$)
- **Description of one observation:** One residential real estate transaction or property listing containing physical structure measurements, legal documentation tier, interior condition, and spatial address components.

### 3.3 Feature Specification Table
| Feature | Type (Numerical / Categorical) | Description & Meaning |
| :--- | :--- | :--- |
| `Address` | Categorical (Spatial Text) | Complete raw listing address (decomposed into `Province_City` and `District`) |
| `Area` | Numerical (Continuous) | Usable floor / land area in square meters ($\text{m}^2$) |
| `Frontage` | Numerical (Continuous) | Width of property facade facing the street in meters ($\text{m}$) |
| `Access Road` | Numerical (Continuous) | Width of alleyway / road leading to property entrance in meters ($\text{m}$) |
| `Floors` | Numerical (Discrete) | Number of building levels / floors ($1 - 9$) |
| `Bedrooms` | Numerical (Discrete) | Number of dedicated bedroom accommodations ($1 - 9$) |
| `Bathrooms` | Numerical (Discrete) | Number of functional bathroom / sanitation facilities ($1 - 9$) |
| `Legal status` | Categorical (Nominal) | Property ownership documentation (`Have certificate`, `Sale contract`, `Unknown`) |
| `Furniture state` | Categorical (Ordinal/Nominal) | Interior furnishing readiness level (`Full`, `Basic`, `Unknown`) |
| `House direction` | Categorical (Nominal) | Compass orientation of main entrance (8 directions, 70.3% missing) |
| `Balcony direction`| Categorical (Nominal) | Compass orientation of primary balcony (8 directions, 82.6% missing) |
| `Price` | Numerical (Continuous) | Market listing selling price in Billion VND (Target variable $y$) |

---

## 4. Application 3 — E-Commerce Customer Behavior & Interest Discovery

### 4.1 Problem Description
- **Real-world Problem:**
  > In modern multi-category e-commerce fashion retail, discovering customer product interests and category affinity is essential for personalized catalog recommendations, targeted marketing campaigns, and catalog navigation. Customer behavior manifests across two modalities: numerical transactional and feedback behavior (age, satisfaction ratings, community feedback helpfulness), and rich unstructured textual feedback (product review body and headline). The objective is to discover and predict a customer's product department interest from their review feedback and behavioral engagement profile.
- **Specific Target Formulation:**
  - **Selected Formulation:** Multi-class Product Department Interest Discovery:
    $$y \in \{\text{'Tops'}, \text{'Dresses'}, \text{'Bottoms'}, \text{'Intimate'}, \text{'Jackets'}\}$$
  - Encoded discrete class target: $y \in \{0, 1, 2, 3, 4\}$.
- **Formal Mathematical Definition:**
  - Multimodal input representation:
    $$\mathbf{x}_i = [\mathbf{x}_{i, \text{tab}} \,\|\, \mathbf{x}_{i, \text{text}}]^T \in \mathbb{R}^{d_{\text{tab}} + d_{\text{text}}}$$
  - Tabular behavioral vector:
    $$\mathbf{x}_{i, \text{tab}} = [x_{\text{age}}, x_{\text{rating}}, x_{\text{recommend}}, x_{\text{positive\_feedback}}, x_{\text{char\_len}}, x_{\text{word\_count}}, x_{\text{upper\_ratio}}, \dots]^T \in \mathbb{R}^{d_{\text{tab}}}$$
  - Text representation:
    $$\text{Review Text} \xrightarrow{\text{Tokenization}} \text{Tokens } [w_1, \dots, w_T] \xrightarrow{\text{Vocabulary } \mathcal{V}} \text{Token IDs } \mathbf{t} \in \mathbb{Z}^T \xrightarrow{\text{TF-IDF / N-gram Vectorizer}} \mathbf{x}_{i, \text{text}} \in \mathbb{R}^{d_{\text{text}}}$$
  - Total feature matrix:
    $$X \in \mathbb{R}^{N \times (d_{\text{tab}} + d_{\text{text}})}, \quad \mathbf{y} \in \{0, 1, 2, 3, 4\}^N$$
  - Objective: Learn hypothesis classifier $f_\theta: \mathbb{R}^{d_{\text{tab}} + d_{\text{text}}} \to \Delta^4$ minimizing categorical cross-entropy loss:
    $$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^4 y_{i,c} \log \hat{P}(y_i = c \mid \mathbf{x}_i; \theta)$$

### 4.2 Kaggle Dataset Details
- **Dataset Name:** Women's Clothing E-Commerce Reviews (`women-clothes.csv`)
- **Kaggle URL:** https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- **Observations ($N$):** 23,486 raw records ($22,510$ complete cleaned text+behavioral records across the 5 primary departments)
- **Attributes / Features ($d$):** 10 input attributes (tabular behavioral + text) + 1 multi-class target (`Department Name`)
- **Target Variable:** `Department Name` (5 classes: `Tops`, `Dresses`, `Bottoms`, `Intimate`, `Jackets`)
- **Description of one observation:** A customer purchase review record containing reviewer age, quantitative star rating, recommendation indicator, community positive feedback count, clothing category metadata, and free-form review headline and text body.

### 4.3 Feature Specification Table
| Feature | Type (Numerical / Categorical / Text) | Description & Meaning |
| :--- | :--- | :--- |
| `Clothing ID` | Categorical (Nominal ID) | Unique integer identifier of the specific clothing item |
| `Age` | Numerical (Continuous / Discrete) | Age of the customer reviewer in years ($18 - 99$) |
| `Title` | Text (Short String) | Headline summary title of the customer review |
| `Review Text` | Text (Unstructured Body) | Detailed free-form textual feedback and product description |
| `Rating` | Numerical (Ordinal) | Customer product rating from $1$ (worst) to $5$ (best) |
| `Recommended IND` | Categorical / Binary | Recommendation flag ($1 = \text{recommends}$, $0 = \text{does not recommend}$) |
| `Positive Feedback Count` | Numerical (Discrete Count) | Number of other customers who found this review helpful ($0 - 122$) |
| `Division Name` | Categorical (Nominal) | High-level commercial apparel division (`General`, `General Petite`, `Intimates`) |
| `Class Name` | Categorical (Nominal) | Detailed product merchandise class (20 classes: `Dresses`, `Knits`, `Blouses`, `Pants`, `Jeans`, etc.) |
| `Review Length` | Numerical (Engineered) | Character count of the cleaned customer review text |
| `Word Count` | Numerical (Engineered) | Total number of whitespace-delimited tokens in review text |
| `Department Name` | Categorical (Target $y$) | Product department representing customer interest (`Tops`, `Dresses`, `Bottoms`, `Intimate`, `Jackets`) |

---

## 5. Connection to Lecture 02: Representation Summary Table

| Application | Raw Data Form | Intermediate Form | Numerical Representation | Model Input Tensor Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | CSV / tabular clinical features | Cleaned DataFrame | Imputed & Standardized feature matrix | $X \in \mathbb{R}^{B \times d}$ |
| **House Price** | CSV / tabular property features | One-hot/Target encoded DataFrame | Standardized feature matrix | $X \in \mathbb{R}^{B \times d}$ |
| **E-Commerce** | CSV transactions + text comments | Tabular vectors + Tokenized text | Concatenated feature matrix / embeddings | $X \in \mathbb{R}^{B \times (d_{\text{tab}} + d_{\text{text}})}$ |

