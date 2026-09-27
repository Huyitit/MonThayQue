# Environment & Reproducibility Guide (`requirements.md`)

This document specifies the software environment, operating system, package dependencies, and step-by-step instructions to reproduce all experiments, pipelines, and web deployments across the three applications in **Assignment 02**.

---

## 1. System & Runtime Specifications

- **Operating System:** Linux (Ubuntu 22.04 LTS / Debian-based or compatible POSIX) / macOS / Windows
- **Python Version:** Python 3.10+ (Recommended: Python 3.11 or 3.12)
- **Package Manager:** `pip` (standard python package installer)
- **Standard Random Seed:** `RANDOM_SEED = 42` (must be fixed across NumPy, Python standard `random`, and scikit-learn for deterministic results)

```python
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

---

## 2. Core Dependencies & Versions

The following libraries form the core stack required across the project:

### A. Data Science & Machine Learning
- `numpy >= 1.24.0`: Tensor & numerical matrix operations
- `pandas >= 2.0.0`: Tabular data loading, cleaning, and inspection
- `scipy >= 1.10.0`: Scientific computations & distributions
- `scikit-learn >= 1.3.0`: Preprocessing pipelines, standard scalers, encoders, classifiers, and regressors
- `joblib >= 1.3.0`: Serialization and deserialization of fitted pipelines and models
- `category_encoders >= 2.6.0`: (Optional) Advanced encoding (TargetEncoder, CatBoostEncoder)

### B. Exploratory Data Analysis & Visualization
- `matplotlib >= 3.7.0`: Plotting distributions, ROC curves, and scatter plots
- `seaborn >= 0.12.0`: Statistical data visualization and correlation heatmaps

### C. Text & Natural Language Processing (Application 3)
- `nltk >= 3.8.0` / `spacy >= 3.6.0`: Text normalization, cleaning, and tokenization
- `scikit-learn`: `TfidfVectorizer`, `CountVectorizer` for text feature extraction

### D. Web Deployment & API Services
- `fastapi >= 0.100.0`: High-performance RESTful API framework
- `uvicorn >= 0.22.0`: ASGI web server implementation
- `pydantic >= 2.0.0`: Input data schema validation and type enforcement
- `flask >= 2.3.0`: (Alternative / Optional) Lightweight web server for template rendering

### E. Development & Interactive Notebooks
- `jupyterlab >= 4.0.0` or `notebook >= 7.0.0`: Interactive experimentation environment
- `ipykernel`: Jupyter kernel runtime

---

## 3. Environment Setup & Step-by-Step Reproduction

### Step 1: Create and Activate a Virtual Environment
From the project root directory (`Assignment_02/`):

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### Step 2: Install Global Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Alternatively, navigate to any specific application folder, e.g. `diabetes/`, and install using `pip install -r requirements.txt`.)*

### Step 3: Reproducing the ML Notebooks
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Navigate to the respective `notebook/` folder (`diabetes/notebook/`, `house_price/notebook/`, or `customer_behavior/notebook/`).
3. Run all cells from top to bottom.
4. Verify that:
   - Data is cleaned and transformed according to the documented representation.
   - The serialized models and preprocessing objects are exported to the corresponding `model/` folder (`model/preprocessor.joblib`, `model/model.joblib`).

### Step 4: Running Web Services
To start a web prediction API (for example, with FastAPI):
```bash
cd diabetes/api
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` in your browser to access the interactive Swagger UI and test `/predict` endpoints.
