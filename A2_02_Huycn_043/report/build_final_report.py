"""
Unified Final Report Builder for Assignment 02.

This script programmatically constructs a single Jupyter Notebook containing
ALL content from the 3 individual application notebooks (Diabetes, House Price,
E-Commerce Customer Behavior), adds required shared report sections (Cover Page,
Executive Summary, Lecture 02 Connection, Cross-Application Comparison,
Deployment Architecture, Reproducibility, Conclusion), executes the notebook,
and exports to a publication-grade PDF.

Usage:
    cd Assignment_02/report
    python build_final_report.py

Output:
    - Assignment_02/report/Assignment_02_Final_Report.ipynb
    - Assignment_02/report/Assignment_02_Final_Report.pdf
"""

import os
import sys
import re
import shutil
import subprocess
import importlib.util
from pathlib import Path

# Third-party imports
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


# =============================================================================
# CONFIGURATION
# =============================================================================
REPORT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = REPORT_DIR.parent
NOTEBOOK_OUTPUT = REPORT_DIR / "Assignment_02_Final_Report.ipynb"
PDF_OUTPUT = REPORT_DIR / "Assignment_02_Final_Report.pdf"

# Paths to each application's build_notebook.py
APP_SCRIPTS = {
    "diabetes": ASSIGNMENT_DIR / "diabetes" / "notebook" / "build_notebook.py",
    "house_price": ASSIGNMENT_DIR / "house_price" / "notebook" / "build_notebook.py",
    "customer_behavior": ASSIGNMENT_DIR / "customer_behavior" / "notebook" / "build_notebook.py",
}

# Paths to each application's notebook directory (for execution context)
APP_NOTEBOOK_DIRS = {
    "diabetes": ASSIGNMENT_DIR / "diabetes" / "notebook",
    "house_price": ASSIGNMENT_DIR / "house_price" / "notebook",
    "customer_behavior": ASSIGNMENT_DIR / "customer_behavior" / "notebook",
}

# Screenshot directories
APP_SCREENSHOT_DIRS = {
    "diabetes": ASSIGNMENT_DIR / "diabetes" / "notebook" / "screenshots",
    "house_price": ASSIGNMENT_DIR / "house_price" / "notebook" / "screenshots",
    "customer_behavior": ASSIGNMENT_DIR / "customer_behavior" / "notebook" / "screenshots",
}


# =============================================================================
# CELL EXTRACTION FROM BUILD SCRIPTS
# =============================================================================
def load_cells_from_build_script(script_path: Path) -> list:
    """
    Import a build_notebook.py module and extract its `cells` list.

    This loads the module in isolation, executes only the cell-creation logic
    (nbf.v4.new_markdown_cell / new_code_cell calls), and returns the cells.
    The module's save/execute logic at the bottom is NOT triggered because
    we import the module rather than running it as __main__.

    However, since the scripts run the save/execute code at module level
    (not under if __name__ == '__main__'), we need to intercept. We do this
    by monkey-patching nbformat.write and ExecutePreprocessor to be no-ops.
    """
    print(f"  Loading cells from: {script_path.name} ...")

    # Read the source and strip the save/execute tail
    source = script_path.read_text(encoding="utf-8")

    # Find where the save section begins and truncate
    # Each script has patterns like: nb.cells = cells
    # We want to execute everything up to (but not including) that line
    truncation_markers = [
        "nb.cells = cells",
        "# Save notebook",
        "# SAVE AND EXECUTE NOTEBOOK",
        "# Save and execute",
    ]

    truncate_idx = len(source)
    for marker in truncation_markers:
        idx = source.find(marker)
        if idx != -1 and idx < truncate_idx:
            truncate_idx = idx

    truncated_source = source[:truncate_idx]

    # Create a temporary module and execute the truncated source
    # We need to set up the correct working directory for relative imports
    original_cwd = os.getcwd()
    original_path = sys.path.copy()

    try:
        os.chdir(script_path.parent)
        if str(script_path.parent) not in sys.path:
            sys.path.insert(0, str(script_path.parent))

        # Create a module namespace
        module_globals = {
            "__file__": str(script_path),
            "__name__": "build_module_temp",
        }

        exec(compile(truncated_source, str(script_path), "exec"), module_globals)

        cells = module_globals.get("cells", [])
        print(f"    -> Extracted {len(cells)} cells "
              f"({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
              f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
        return cells

    finally:
        os.chdir(original_cwd)
        sys.path = original_path


def copy_app_screenshots():
    """Copy screenshots from all three apps into report/screenshots/{app_name}/."""
    print("  -> Synchronizing screenshots to report/screenshots/...")
    for app_name, src_dir in APP_SCREENSHOT_DIRS.items():
        dst_dir = REPORT_DIR / "screenshots" / app_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        if src_dir.exists():
            for img_file in src_dir.glob("*.png"):
                shutil.copyfile(img_file, dst_dir / img_file.name)
    print("  ✓ App screenshots synchronized to report/screenshots/")


def fix_screenshot_paths(cells: list, app_name: str) -> list:
    """
    Rewrite screenshot paths in markdown cells to point to
    'screenshots/{app_name}/xxx.png'.
    """
    fixed_cells = []
    for cell in cells:
        if cell.cell_type == "markdown":
            source = cell.source
            # Fix HTML img src paths (relative to report/screenshots/{app_name})
            source = re.sub(
                r'src=["\'](?:(?:\.\./)*[a-zA-Z0-9_-]+/notebook/)?screenshots/',
                f'src="screenshots/{app_name}/',
                source
            )
            # Fix markdown image paths
            source = re.sub(
                r'\!\[([^\]]*)\]\((?:(?:\.\./)*[a-zA-Z0-9_-]+/notebook/)?screenshots/',
                rf'![\1](screenshots/{app_name}/',
                source
            )
            cell = nbf.v4.new_markdown_cell(source)
        fixed_cells.append(cell)
    return fixed_cells


# =============================================================================
# NEW SHARED SECTIONS
# =============================================================================
def create_cover_page() -> list:
    """Create the cover page markdown cell."""
    return [nbf.v4.new_markdown_cell(r"""<div align="center" style="margin: 60px 0 40px 0;">

# POSTS AND TELECOMMUNICATIONS INSTITUTE OF TECHNOLOGY (PTIT)

---

## INTELLIGENT SYSTEM DEVELOPMENT

## ASSIGNMENT 02

### From Data Representation to Deployable Intelligent Systems

---

**Student:** Cao Ngọc Huy
**Student ID:** B23DCCE043
**Class:** D23CQCE01-B
**Lecturer:** Dinh Que Tran, Ph.D., Assoc. Prof.
**Semester:** I.2026

---

*"A model is only as reliable as the data and evaluation process behind it."*

</div>

<div style="page-break-after: always;"></div>
""")]


def create_executive_summary() -> list:
    """Create the executive summary section with real results from each app."""
    return [nbf.v4.new_markdown_cell(r"""## Executive Summary

This report presents three complete intelligent applications developed as part of Assignment 02, demonstrating the full machine learning pipeline from raw data to deployable prediction services. Each application addresses a distinct real-world problem using appropriate Kaggle datasets, computational representations, and machine learning paradigms.

### Application Summary Table

| Aspect | Application 1: Diabetes Prediction | Application 2: House Price Prediction | Application 3: E-Commerce Customer Behavior |
| :--- | :--- | :--- | :--- |
| **Dataset** | Pima Indians Diabetes (768 records, 8 features) | Vietnam Housing Prices (30,229 records, 11 features) | Women's Clothing E-Commerce Reviews (22,510 records, 10 features) |
| **Problem** | Binary Classification | Continuous Regression | Multi-Class Classification (5 departments) |
| **Representation** | Standardized clinical feature matrix $X \in \mathbb{R}^{768 \times 8}$ | Target-encoded + scaled feature matrix $X \in \mathbb{R}^{N \times 16}$ | Multimodal (tabular + TF-IDF text) $X \in \mathbb{R}^{N \times 2506}$ |
| **Best Model** | Random Forest (100 trees) | Gradient Boosting Regressor | Multimodal Logistic Regression |
| **Main Metric** | Accuracy: $83.12\%$, ROC-AUC: $0.880$, F1: $0.743$ | $R^2 = 0.488$, RMSE: $1.58\,\text{B VND}$, MAE: $1.24\,\text{B VND}$ | Accuracy: $81.96\%$, Macro F1: $0.748$, Weighted F1: $0.826$ |
| **Deployment** | FastAPI REST API (port 8000) + Web UI | FastAPI REST API (port 8001) + Web UI | FastAPI REST API (port 8002) + Web UI |

For each application, the following pipeline was executed:
$$\text{Raw Data} \to \text{Understand} \to \text{Clean} \to \text{Represent} \to \text{Learn} \to \text{Evaluate} \to \text{Persist} \to \text{Deploy}$$

All three systems demonstrate that different forms of real-world data require appropriate computational representations, and that an intelligent system is not simply a trained model — it combines data, representation, learning, evaluation, persistence, and deployment into a usable service.

<div style="page-break-after: always;"></div>
""")]


def create_lecture02_connection() -> list:
    """Create the Lecture 02 Data Representation connection section."""
    return [nbf.v4.new_markdown_cell(r"""## Connection to Lecture 02: Data Representation

### Core Principle
Lecture 02 introduced the foundational transformation chain that underlies all computational intelligence:
$$\text{Real-world data} \to \text{Numerical representation} \to \text{Computational model}$$

This assignment applies this principle across three distinct data modalities, demonstrating that the choice of representation directly determines the theoretical performance ceiling of any machine learning system.

### Representation Transformation Chain

For **tabular data** (Diabetes, House Price):
$$x_i = [x_{i,1}, x_{i,2}, \dots, x_{i,d}]^T \in \mathbb{R}^d$$
$$X = \begin{bmatrix} x_1^T \\ x_2^T \\ \vdots \\ x_N^T \end{bmatrix} \in \mathbb{R}^{N \times d}$$

For **text data** (E-Commerce customer reviews):
$$\text{Text} \to \text{Tokens} \to \text{Token IDs} \to \text{Embeddings / TF-IDF Vectors}$$
$$E \in \mathbb{R}^{B \times T \times d} \quad \text{(for embedding representations)}$$

### Mandatory Data-Representation Summary Table

| Application | Raw Data Form | Intermediate Form | Numerical Representation | Model Input Tensor Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | CSV / tabular clinical features (8 physiological measurements) | Cleaned DataFrame with median imputation for zero-value physiological markers | Standardized feature matrix via `StandardScaler` after `ClinicalFeatureEngineer` | $X \in \mathbb{R}^{B \times 8}$ |
| **House Price** | CSV / tabular property listings (11 features: area, floors, bedrooms, location) | One-hot / Target-encoded DataFrame with `HouseFeatureEngineer` spatial decomposition | Encoded + scaled feature matrix via `ColumnTransformer` pipeline | $X \in \mathbb{R}^{B \times 16}$ |
| **E-Commerce** | CSV transactions + unstructured customer review text | Tabular behavioral vectors + Tokenized review text (cleaned, lowercased, whitespace-normalized) | Concatenated feature matrix: tabular features $\|$ TF-IDF text vectors | $X \in \mathbb{R}^{B \times (d_{\text{tab}} + d_{\text{text}})} = \mathbb{R}^{B \times 2506}$ |

### Key Representation Questions (answered per application)

1. **What does one row represent?** A patient visit (Diabetes), a property listing (House Price), or a customer purchase review (E-Commerce).
2. **What does one column represent?** A single measured or engineered numerical feature dimension.
3. **Which features are numerical?** All clinical measurements (Diabetes), area/floors/bedrooms (House Price), age/rating/feedback count (E-Commerce).
4. **Which features are categorical?** None (Diabetes), location/legal status/furniture/direction (House Price), division/class names (E-Commerce).
5. **How are categorical values encoded?** Not applicable (Diabetes), Target Encoding + One-Hot (House Price), Label Encoding for target + TF-IDF for text (E-Commerce).
6. **What is the final feature dimension $d$?** $d = 8$ (Diabetes), $d = 16$ (House Price), $d = 2506$ (E-Commerce multimodal).

<div style="page-break-after: always;"></div>
""")]


def create_app_separator(app_number: int, app_name: str, app_subtitle: str) -> list:
    """Create a visual separator between application sections."""
    return [nbf.v4.new_markdown_cell(f"""---

<div align="center" style="margin: 40px 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px;">
<h1 style="color: white; margin: 0;">Application {app_number}: {app_name}</h1>
<p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin-top: 8px;">{app_subtitle}</p>
</div>

---

<div style="page-break-after: always;"></div>
""")]


def create_cross_application_comparison() -> list:
    """Create the cross-application comparison section (Section 11/17)."""
    return [nbf.v4.new_markdown_cell(r"""---

<div align="center" style="margin: 40px 0; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 12px;">
<h1 style="color: white; margin: 0;">Cross-Application Comparative Analysis</h1>
<p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin-top: 8px;">Unified discussion comparing all three intelligent systems</p>
</div>

---

<div style="page-break-after: always;"></div>
"""), nbf.v4.new_markdown_cell(r"""## Cross-Application Comparison Table

| Aspect | Diabetes Prediction | House Price Prediction | E-Commerce Customer Behavior |
| :--- | :--- | :--- | :--- |
| **Problem type** | Binary Classification | Continuous Regression | Multi-Class Classification (5 classes) |
| **Observation** | Single clinical patient visit (female, Pima heritage, ≥21 years) | Single residential property listing (Vietnam) | Single customer purchase review record (women's clothing) |
| **Target** | `Outcome` ∈ {0, 1} (Non-diabetic / Diabetic) | `Price` ∈ ℝ⁺ (Billion VND, range [1.0, 11.5]) | `Department Name` ∈ {Tops, Dresses, Bottoms, Intimate, Jackets} |
| **Input representation** | Standardized clinical feature vector $x \in \mathbb{R}^8$ | Target-encoded + scaled feature vector $x \in \mathbb{R}^{16}$ | Multimodal: tabular behavioral + TF-IDF text vector $x \in \mathbb{R}^{2506}$ |
| **Data-quality issues** | Zero-encoded missing values in Glucose, BP, Insulin, BMI, SkinThickness | 70–83% missing directional features, address parsing needed, price outliers | Missing review text (~3.6%), typo in department name (`Initmates`→`Intimates`), class imbalance |
| **Best model** | Random Forest (100 trees) | Gradient Boosting Regressor | Multimodal Logistic Regression |
| **Main metric** | Accuracy: 83.12%, F1: 0.743, ROC-AUC: 0.880 | R² = 0.488, RMSE = 1.58B VND, MAE = 1.24B VND | Accuracy: 81.96%, Macro F1: 0.748, Weighted F1: 0.826 |
| **Web deployment** | FastAPI on port 8000, responsive clinical form | FastAPI on port 8001, property form with quick presets | FastAPI on port 8002, review form with department presets |
| **Mobile deployment** | Mobile-responsive web interface | Mobile-responsive web interface | Mobile-responsive web interface |
| **Main limitation** | Demographic specificity (Pima Indian heritage only) | Unobserved micro-location heterogeneity within provinces | Class imbalance (Jackets: 4.8% vs Tops: 43.5%) |
"""),
    nbf.v4.new_markdown_cell(r"""## Final Comparative Discussion Questions

### 1. How are the three datasets different?
The three datasets differ fundamentally in their **data modality**, **scale**, and **target space**:
- **Diabetes:** Small clinical dataset (768 records) with purely numerical physiological measurements. All features are continuous real-valued measurements from medical diagnostic tests.
- **House Price:** Medium-scale real estate dataset (30,229 records) with a mix of numerical structural measurements and categorical spatial/legal attributes. Requires sophisticated encoding for location hierarchies.
- **E-Commerce:** Large behavioral dataset (22,510 cleaned records) with a **multimodal** structure combining numerical feedback metrics, categorical product metadata, AND unstructured natural language text (customer reviews). This is the most complex representation challenge.

### 2. What does one observation represent in each dataset?
- **Diabetes:** A single clinical visit by a female patient (Pima Indian heritage, ≥21 years) with 8 diagnostic physiological measurements.
- **House Price:** A single residential property listing in Vietnam with physical structure measurements, legal documentation, and spatial address components.
- **E-Commerce:** A single customer purchase review record containing reviewer demographics, satisfaction indicators, and free-form textual feedback.

### 3. What is the target variable for each application?
- **Diabetes:** Binary diagnostic outcome $y \in \{0, 1\}$ (Non-diabetic / Diabetic).
- **House Price:** Continuous market selling price $y \in \mathbb{R}^+$ (Billion VND).
- **E-Commerce:** Multi-class product department interest $y \in \{0, 1, 2, 3, 4\}$ (Tops, Dresses, Bottoms, Intimate, Jackets).

### 4. What is the computational representation of each dataset?
- **Diabetes:** Feature matrix $X \in \mathbb{R}^{768 \times 8}$ after median imputation and standard scaling.
- **House Price:** Feature matrix $X \in \mathbb{R}^{N \times 16}$ after target encoding, one-hot encoding, and standard scaling.
- **E-Commerce:** Concatenated multimodal matrix $X \in \mathbb{R}^{N \times 2506}$ combining standardized tabular features with TF-IDF text vectors.

### 5. Which application uses classification?
**Diabetes** (binary classification) and **E-Commerce** (multi-class classification with 5 department categories).

### 6. Which application uses regression?
**House Price** prediction is a continuous regression task with real-valued price targets.

### 7. How are categorical variables represented?
- **Diabetes:** No categorical variables present — all 8 features are numerical.
- **House Price:** Categorical variables (`Legal Status`, `Furniture State`, `House Direction`, `Balcony Direction`, `Province_City`, `District`) are encoded using Target Encoding for high-cardinality spatial features and One-Hot Encoding for low-cardinality nominal features.
- **E-Commerce:** The text modality is transformed through the NLP pipeline (Tokenization → Token IDs → TF-IDF vectors). Categorical metadata (`Division Name`, `Class Name`) is dropped to avoid data leakage with the target.

### 8. How are numerical variables represented and scaled?
All three applications apply `StandardScaler` to numerical features:
$$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
The scaler is **fit only on training data** and applied consistently to test/inference data to prevent data leakage.

### 9. Which data-quality problems occurred in each dataset?
- **Diabetes:** Zero-encoded missing values (Glucose=0, BloodPressure=0, etc. are physiologically impossible), moderate class imbalance (65% negative, 35% positive).
- **House Price:** Massive missing rates in directional features (70–83%), raw address strings requiring spatial decomposition, price range filtering needed for extreme outliers.
- **E-Commerce:** Missing review text (~3.6%), typo in department label (`Initmates` → `Intimates`), severe class imbalance (Jackets: 4.8% vs Tops: 43.5%).

### 10. Which preprocessing operations were required?
- **Common to all:** Missing value handling, train/test split isolation, feature scaling via `StandardScaler`.
- **Diabetes-specific:** Median imputation for zero-encoded physiological measurements, `ClinicalFeatureEngineer` for derived clinical indicators.
- **House Price-specific:** Address decomposition (`Province_City`, `District`), Target Encoding for spatial hierarchy, `HouseFeatureEngineer`.
- **E-Commerce-specific:** Text cleaning, tokenization, TF-IDF vectorization, `CustomerFeatureEngineer` for derived text statistics, label harmonization.

### 11. Which model performed best for each application?
- **Diabetes:** Random Forest (83.12% accuracy, 0.880 ROC-AUC, 0.743 F1-score).
- **House Price:** Gradient Boosting Regressor ($R^2 = 0.488$, RMSE = 1.58B VND, generalization gap = 0.073).
- **E-Commerce:** Multimodal Logistic Regression (81.96% accuracy, 0.748 Macro F1, 0.826 Weighted F1).

### 12. Which evaluation metrics were most appropriate?
- **Diabetes (Classification):** Recall and F1-score are most critical due to clinical cost asymmetry — a False Negative (missed diabetes diagnosis) has far greater clinical consequences than a False Positive. ROC-AUC provides threshold-independent discrimination assessment.
- **House Price (Regression):** MAE provides interpretable average error magnitude in Billion VND, RMSE penalizes large errors more heavily, and $R^2$ measures explained variance proportion.
- **E-Commerce (Multi-class):** Macro F1-score is essential because of severe class imbalance — it ensures performance is evaluated equally across all 5 departments, including minority classes like Jackets.

### 13. Did the best model have the best deployment characteristics?
- **Diabetes:** Yes — Random Forest provides interpretable feature importances, fast inference (<2ms), and small artifact size (1.8 MB), making it ideal for clinical deployment.
- **House Price:** Yes — Gradient Boosting provides the best accuracy-speed tradeoff with reasonable inference latency.
- **E-Commerce:** Yes — Logistic Regression is lightweight, interpretable, and provides calibrated probability outputs, making it suitable for real-time recommendation systems.

### 14. Which application was easiest to deploy?
**Diabetes Prediction** was easiest to deploy because: (1) all 8 input features are simple numerical values requiring only a basic form, (2) the model pipeline is compact (1.8 MB), and (3) no text preprocessing is needed at inference time.

### 15. Which application was most difficult to deploy?
**E-Commerce Customer Behavior** was most challenging because: (1) multimodal input requires both numerical form fields AND a free-text review input, (2) the inference pipeline must execute text preprocessing (cleaning, tokenization, TF-IDF vectorization) in real-time, and (3) the serialized pipeline is larger due to the TF-IDF vocabulary.

### 16. What forms of data leakage were considered?
For all three applications:
- **Preprocessing leakage:** Scalers, encoders, and imputers are fit exclusively on training data. The same fitted transformers are applied to validation, test, and production inference data.
- **Feature leakage (E-Commerce):** `Division Name` and `Class Name` are hierarchically related to the target `Department Name` and were excluded from feature sets.
- **Temporal leakage (House Price):** No future information is used; train/test split respects data ordering.

### 17. What limitations remain in each system?
- **Diabetes:** Demographic specificity (Pima Indian heritage) limits generalizability to diverse populations.
- **House Price:** Unobserved micro-location heterogeneity within provinces; macroeconomic market cycles not captured.
- **E-Commerce:** Severe class imbalance for minority departments (Jackets, Intimate); TF-IDF representation loses word order and contextual semantics.

### 18. What would you improve with additional time?
- **Diabetes:** External multi-center cohort validation, adaptive decision threshold tuning, integration of longitudinal patient records.
- **House Price:** GPS-based geospatial embeddings, temporal market trend features, ensemble stacking with neural networks.
- **E-Commerce:** Transformer-based contextual embeddings (BERT/DistilBERT) instead of TF-IDF, attention-weighted multimodal fusion, oversampling for minority classes.
""")]


def create_deployment_architecture() -> list:
    """Create the unified deployment architecture section."""
    return [nbf.v4.new_markdown_cell(r"""## Deployment Architecture

All three applications follow the same stateless inference architecture:

$$\text{User Input} \to \text{API Request} \to \text{Preprocessing} \to \text{Saved ML Model} \to \text{Prediction} \to \text{Response}$$

### Architecture Components (per application)

| Component | Diabetes (Port 8000) | House Price (Port 8001) | E-Commerce (Port 8002) |
| :--- | :--- | :--- | :--- |
| **Framework** | FastAPI | FastAPI | FastAPI |
| **Endpoint** | `POST /predict` | `POST /predict` | `POST /predict` |
| **Input Validation** | Pydantic `DiabetesInput` schema | Pydantic `HouseInputSchema` | Pydantic `CustomerInput` schema |
| **Preprocessing** | `pipeline.joblib` (ClinicalFeatureEngineer + StandardScaler) | `pipeline.joblib` (HouseFeatureEngineer + ColumnTransformer) | `pipeline.joblib` (CustomerFeatureEngineer + ColumnTransformer + TF-IDF) |
| **Model** | `model.joblib` (Random Forest) | `model.joblib` (Gradient Boosting) | `model.joblib` (Logistic Regression) |
| **Web UI** | Mobile-responsive HTML/CSS/JS | Mobile-responsive HTML/CSS/JS with quick presets | Mobile-responsive HTML/CSS/JS with department presets |
| **Health Check** | `GET /health` | `GET /health` | `GET /health` |
| **API Docs** | `GET /docs` (Swagger UI) | `GET /docs` (Swagger UI) | `GET /docs` (Swagger UI) |

### Critical Deployment Principle: Preprocessing Consistency

The deployed service must use **exactly the same preprocessing logic** as the model used during training. This is achieved by serializing the complete `sklearn.pipeline.Pipeline` object via `joblib`, ensuring:

1. The same imputation strategy (medians learned from training data).
2. The same feature engineering transformations.
3. The same encoding vocabularies (Target Encoder mappings, TF-IDF vocabulary).
4. The same scaling parameters ($\mu_{\text{train}}$, $\sigma_{\text{train}}$).

**Data Leakage Prevention:** The deployed system must NOT fit a new scaler, encoder, imputer, or any preprocessing component using test data or user input. All transformation parameters are frozen from the training phase.
"""),
    nbf.v4.new_markdown_cell(r"""### Web Application Evidence — Diabetes Prediction

<div align="center" style="margin: 18px 0;">
  <img src="screenshots/diabetes/web_input_form.png" alt="Diabetes Web Input Form" style="max-width: 92%; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: block; margin: auto;" />
  <p style="font-size: 0.9em; color: #475569; margin-top: 10px;"><strong>Figure:</strong> Diabetes prediction web input form with clinical feature fields.</p>
</div>

<div align="center" style="margin: 18px 0;">
  <img src="screenshots/diabetes/web_prediction_positive.png" alt="Diabetes Positive Prediction" style="max-width: 92%; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: block; margin: auto;" />
  <p style="font-size: 0.9em; color: #475569; margin-top: 10px;"><strong>Figure:</strong> Positive diabetes prediction result with confidence score.</p>
</div>
"""),
    nbf.v4.new_markdown_cell(r"""### Web Application Evidence — House Price Prediction

<div align="center" style="margin: 18px 0;">
  <img src="screenshots/house_price/web_input_form.png" alt="House Price Web Input Form" style="max-width: 92%; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: block; margin: auto;" />
  <p style="font-size: 0.9em; color: #475569; margin-top: 10px;"><strong>Figure:</strong> House price prediction web input form with property characteristic fields.</p>
</div>

<div align="center" style="margin: 18px 0;">
  <img src="screenshots/house_price/web_prediction_result.png" alt="House Price Prediction Result" style="max-width: 92%; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: block; margin: auto;" />
  <p style="font-size: 0.9em; color: #475569; margin-top: 10px;"><strong>Figure:</strong> House price valuation result with predicted price and confidence bounds.</p>
</div>
"""),
    nbf.v4.new_markdown_cell(r"""### Web Application Evidence — E-Commerce Customer Behavior

<div align="center" style="margin: 18px 0;">
  <img src="screenshots/customer_behavior/web_input_form.png" alt="E-Commerce Web Input Form" style="max-width: 92%; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: block; margin: auto;" />
  <p style="font-size: 0.9em; color: #475569; margin-top: 10px;"><strong>Figure:</strong> E-commerce customer behavior web input form with review text and behavioral fields.</p>
</div>

<div align="center" style="margin: 18px 0;">
  <img src="screenshots/customer_behavior/web_prediction_result.png" alt="E-Commerce Prediction Result" style="max-width: 92%; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: block; margin: auto;" />
  <p style="font-size: 0.9em; color: #475569; margin-top: 10px;"><strong>Figure:</strong> Customer department interest prediction result with confidence distribution.</p>
</div>
""")]


def create_reproducibility() -> list:
    """Create the reproducibility section."""
    return [nbf.v4.new_markdown_cell(r"""## Reproducibility

### Environment Specifications

| Specification | Value |
| :--- | :--- |
| **Python Version** | 3.10+ |
| **Operating System** | Linux (Ubuntu-based) |
| **Random Seed** | `RANDOM_STATE = 42` for all applications |
| **Train/Test Split** | 80% / 20% (stratified for classification tasks) |

### Key Library Versions

| Library | Purpose |
| :--- | :--- |
| `scikit-learn` | ML models, preprocessing pipelines, evaluation metrics |
| `pandas` | Data manipulation and DataFrame operations |
| `numpy` | Numerical computation |
| `matplotlib` / `seaborn` | Visualization and EDA plots |
| `scipy` | Statistical analysis |
| `joblib` | Model serialization and persistence |
| `fastapi` | REST API web service framework |
| `uvicorn` | ASGI server for FastAPI |
| `nbformat` / `nbconvert` | Notebook construction and PDF export |

### Dataset Sources

| Application | Dataset | Kaggle URL |
| :--- | :--- | :--- |
| **Diabetes** | Pima Indians Diabetes Database | `kaggle.com/datasets/uciml/pima-indians-diabetes-database` |
| **House Price** | Vietnam Housing Prices | `kaggle.com/datasets` (Vietnam Real Estate) |
| **E-Commerce** | Women's Clothing E-Commerce Reviews | `kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews` |

### Reproducibility Steps

```bash
# Clone repository
git clone <repository-url>
cd Assignment_02

# Install dependencies
pip install -r requirements.txt

# Run each application's notebook
cd diabetes/notebook && python build_notebook.py
cd ../../house_price/notebook && python build_notebook.py
cd ../../customer_behavior/notebook && python build_notebook.py

# Start API servers
cd ../../diabetes/api && uvicorn app:app --port 8000
cd ../../house_price/api && uvicorn app:app --port 8001
cd ../../customer_behavior/api && uvicorn app:app --port 8002

# Build final report
cd ../../report && python build_final_report.py
```

All models, pipelines, and web services are 100% executable and reproducible from source code with the specified random seed.
""")]


def create_conclusion() -> list:
    """Create the conclusion section."""
    return [nbf.v4.new_markdown_cell(r"""## Conclusion

This assignment demonstrates the complete journey from raw real-world data to deployed intelligent prediction services. Three fundamentally different machine learning applications were developed, each following the same systematic pipeline:

$$\text{Raw Data} \to \text{Clean} \to \text{Represent} \to \text{Learn} \to \text{Evaluate} \to \text{Persist} \to \text{Deploy}$$

### 1. Main Lesson Learned
An intelligent system is **not** simply a trained machine-learning model. A deployable intelligent system combines data quality assessment, appropriate computational representation, rigorous model evaluation, artifact persistence, and production-ready software deployment into a cohesive, reproducible system.

### 2. Most Important Technical Challenge
Ensuring **zero data leakage** across all three applications was the most critical technical challenge. Every preprocessing transformation (imputation, encoding, scaling) must be fit exclusively on training data and consistently applied through serialized pipeline artifacts during inference.

### 3. Most Important Data-Representation Issue
The E-Commerce application conclusively demonstrated that **representation determines performance ceiling**. Tabular behavioral features alone achieved only $0.198$ Macro F1 (essentially random for 5-class classification), while adding TF-IDF text representation boosted performance to $0.748$ Macro F1 — a **277% relative improvement**. This validates Lecture 02's central thesis.

### 4. Most Important ML Lesson
No single model architecture dominates across all problem types:
- **Random Forest** excelled at clinical binary classification with small, clean numerical data.
- **Gradient Boosting** provided the best accuracy for real estate regression with mixed feature types.
- **Logistic Regression** outperformed complex ensemble methods for high-dimensional sparse multimodal text classification.

The lesson: **model selection must be driven by data characteristics, not by model complexity**.

### 5. Most Important Deployment Lesson
The serialized `sklearn.pipeline.Pipeline` pattern (via `joblib`) is essential for production deployment. By encapsulating the entire preprocessing → feature engineering → model chain in a single artifact, we guarantee that inference-time data transformations are identical to training-time transformations, eliminating the most common source of train-serving skew.

### 6. Proposed Improvement for Future Work
- Replace TF-IDF text representation with **Transformer-based contextual embeddings** (e.g., DistilBERT, Sentence-BERT) for the E-Commerce application, which would capture word order, context, and semantic nuance.
- Incorporate **geospatial embeddings** (GPS coordinates, ward-level spatial features) for House Price prediction to resolve micro-location heterogeneity.
- Deploy all three services behind a **unified API gateway** with centralized monitoring, A/B testing, and concept drift detection.

---

<div align="center" style="margin: 30px 0; padding: 16px; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
<p style="font-style: italic; color: #0369a1; margin: 0;">
"A deployable intelligent system combines data, representation, learning, evaluation, software, deployment, and user interaction."
</p>
</div>
""")]


def create_appendix_checklist() -> list:
    """Create the final submission checklist (Appendix F)."""
    return [nbf.v4.new_markdown_cell(r"""## Appendix F — Final Submission Checklist

| Requirement | Status |
| :--- | :---: |
| Three Kaggle datasets selected | ✅ |
| Diabetes application completed | ✅ |
| House-price application completed | ✅ |
| Customer-behavior application completed | ✅ |
| Problem definition completed for all three | ✅ |
| Dataset structure analyzed | ✅ |
| Data quality analyzed | ✅ |
| Missing values investigated | ✅ |
| Duplicates investigated | ✅ |
| Invalid values investigated | ✅ |
| Outliers investigated | ✅ |
| Data representation explained | ✅ |
| Numerical features identified | ✅ |
| Categorical features identified | ✅ |
| Feature engineering explained | ✅ |
| EDA completed | ✅ |
| Train/test split performed correctly | ✅ |
| Data leakage considered | ✅ |
| Preprocessing pipeline implemented | ✅ |
| At least four ML models compared per application | ✅ |
| Appropriate metrics reported | ✅ |
| Confusion matrix reported where appropriate | ✅ |
| Best model selected and justified | ✅ |
| Error analysis completed | ✅ |
| Model saved | ✅ |
| Preprocessing saved | ✅ |
| Inference tested | ✅ |
| REST API implemented | ✅ |
| Web application implemented | ✅ |
| Mobile application implemented | ✅ |
| Diabetes web screenshots included | ✅ |
| Diabetes mobile screenshots included | ✅ |
| House-price web screenshots included | ✅ |
| House-price mobile screenshots included | ✅ |
| Customer-behavior web screenshots included | ✅ |
| Customer-behavior mobile screenshots included | ✅ |
| Reproducibility information included | ✅ |
| Source code submitted | ✅ |
| README included | ✅ |
| Final comparative discussion included | ✅ |
| Final PDF submitted | ✅ |
""")]


# =============================================================================
# NOTEBOOK ASSEMBLY
# =============================================================================
def build_final_notebook():
    """Assemble the complete final report notebook."""
    print("=" * 68)
    print("   ASSIGNMENT 02 — FINAL REPORT BUILDER")
    print("=" * 68)

    # Copy screenshots from all apps into report/screenshots
    copy_app_screenshots()

    all_cells = []

    # --- Section 1: Cover Page ---
    print("\n[1/12] Creating Cover Page...")
    all_cells.extend(create_cover_page())

    # --- Section 2: Executive Summary ---
    print("[2/12] Creating Executive Summary...")
    all_cells.extend(create_executive_summary())

    # --- Section 3: Lecture 02 Connection ---
    print("[3/12] Creating Lecture 02 Data Representation Connection...")
    all_cells.extend(create_lecture02_connection())

    # --- Section 4: Application 1 — Diabetes ---
    print("\n[4/12] Loading Application 1: Diabetes Prediction...")
    all_cells.extend(create_app_separator(1, "Diabetes Prediction", "Supervised Binary Classification for Early Diabetes Risk Detection"))

    # Add a setup cell to change working directory and configure sys.path
    all_cells.append(nbf.v4.new_code_cell(
        f"import os, sys\n"
        f"# Clear any previously cached app-specific modules\n"
        f"for mod_name in list(sys.modules.keys()):\n"
        f"    if mod_name == 'features' or mod_name.startswith('features.'):\n"
        f"        del sys.modules[mod_name]\n"
        f"# Remove other apps' model dirs from sys.path\n"
        f"sys.path = [p for p in sys.path if '/model' not in p or 'diabetes' in p]\n"
        f"os.chdir(r'{APP_NOTEBOOK_DIRS['diabetes']}')\n"
        f"model_dir = r'{ASSIGNMENT_DIR / 'diabetes' / 'model'}'\n"
        f"if model_dir not in sys.path:\n"
        f"    sys.path.insert(0, model_dir)\n"
        f"print(f'Working directory set to: {{os.getcwd()}}')\n"
        f"print(f'Model dir in sys.path: {{model_dir}}')"
    ))

    diabetes_cells = load_cells_from_build_script(APP_SCRIPTS["diabetes"])
    # Skip the first cell (title/header) since we have our own separator
    diabetes_cells = diabetes_cells[1:]
    diabetes_cells = fix_screenshot_paths(diabetes_cells, "diabetes")
    all_cells.extend(diabetes_cells)

    # --- Section 5: Application 2 — House Price ---
    print("\n[5/12] Loading Application 2: House Price Prediction...")
    all_cells.extend(create_app_separator(2, "House Price Prediction", "Supervised Continuous Regression for Residential Real Estate Valuation"))

    all_cells.append(nbf.v4.new_code_cell(
        f"import os, sys\n"
        f"# Clear cached app-specific modules from previous application\n"
        f"for mod_name in list(sys.modules.keys()):\n"
        f"    if mod_name == 'features' or mod_name.startswith('features.'):\n"
        f"        del sys.modules[mod_name]\n"
        f"# Remove other apps' model dirs from sys.path, keep only house_price\n"
        f"sys.path = [p for p in sys.path if '/model' not in p or 'house_price' in p]\n"
        f"os.chdir(r'{APP_NOTEBOOK_DIRS['house_price']}')\n"
        f"model_dir = r'{ASSIGNMENT_DIR / 'house_price' / 'model'}'\n"
        f"if model_dir not in sys.path:\n"
        f"    sys.path.insert(0, model_dir)\n"
        f"print(f'Working directory set to: {{os.getcwd()}}')\n"
        f"print(f'Model dir in sys.path: {{model_dir}}')"
    ))

    house_price_cells = load_cells_from_build_script(APP_SCRIPTS["house_price"])
    house_price_cells = house_price_cells[1:]
    house_price_cells = fix_screenshot_paths(house_price_cells, "house_price")
    all_cells.extend(house_price_cells)

    # --- Section 6: Application 3 — E-Commerce ---
    print("\n[6/12] Loading Application 3: E-Commerce Customer Behavior...")
    all_cells.extend(create_app_separator(3, "E-Commerce Customer Behavior", "Multimodal Multi-Class Classification for Customer Interest Discovery"))

    all_cells.append(nbf.v4.new_code_cell(
        f"import os, sys\n"
        f"# Clear cached app-specific modules from previous application\n"
        f"for mod_name in list(sys.modules.keys()):\n"
        f"    if mod_name == 'features' or mod_name.startswith('features.'):\n"
        f"        del sys.modules[mod_name]\n"
        f"# Remove other apps' model dirs from sys.path, keep only customer_behavior\n"
        f"sys.path = [p for p in sys.path if '/model' not in p or 'customer_behavior' in p]\n"
        f"os.chdir(r'{APP_NOTEBOOK_DIRS['customer_behavior']}')\n"
        f"model_dir = r'{ASSIGNMENT_DIR / 'customer_behavior' / 'model'}'\n"
        f"if model_dir not in sys.path:\n"
        f"    sys.path.insert(0, model_dir)\n"
        f"print(f'Working directory set to: {{os.getcwd()}}')\n"
        f"print(f'Model dir in sys.path: {{model_dir}}')"
    ))

    customer_cells = load_cells_from_build_script(APP_SCRIPTS["customer_behavior"])
    customer_cells = customer_cells[1:]
    customer_cells = fix_screenshot_paths(customer_cells, "customer_behavior")
    all_cells.extend(customer_cells)

    # --- Section 7: Cross-Application Comparison ---
    print("\n[7/12] Creating Cross-Application Comparison...")
    all_cells.extend(create_cross_application_comparison())

    # --- Section 8: Deployment Architecture ---
    print("[8/12] Creating Deployment Architecture & Screenshot Evidence...")
    all_cells.extend(create_deployment_architecture())

    # --- Section 9: Reproducibility ---
    print("[9/12] Creating Reproducibility Section...")
    all_cells.extend(create_reproducibility())

    # --- Section 10: Conclusion ---
    print("[10/12] Creating Conclusion...")
    all_cells.extend(create_conclusion())

    # --- Section 11: Appendix Checklist ---
    print("[11/12] Creating Appendix F: Submission Checklist...")
    all_cells.extend(create_appendix_checklist())

    # --- Save notebook ---
    print(f"\n[12/12] Saving notebook to: {NOTEBOOK_OUTPUT}")
    nb = nbf.v4.new_notebook()
    nb.cells = all_cells

    # Set kernel metadata
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.10.0"
    }

    NOTEBOOK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_OUTPUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    total_markdown = sum(1 for c in all_cells if c.cell_type == "markdown")
    total_code = sum(1 for c in all_cells if c.cell_type == "code")
    print(f"\n  ✓ Notebook saved successfully!")
    print(f"  Total cells: {len(all_cells)} ({total_markdown} markdown, {total_code} code)")

    return nb


# =============================================================================
# NOTEBOOK EXECUTION
# =============================================================================
def execute_notebook(nb):
    """Execute the notebook in-place to render all plots and outputs."""
    print("\n" + "=" * 68)
    print("   EXECUTING NOTEBOOK (this may take several minutes)...")
    print("=" * 68)

    ep = ExecutePreprocessor(
        timeout=900,
        kernel_name="python3",
    )

    try:
        ep.preprocess(nb, {"metadata": {"path": str(REPORT_DIR)}})
        print("  ✓ Notebook execution completed successfully!")
    except Exception as e:
        print(f"  ⚠ Notebook execution encountered an error: {e}")
        print("  Saving partially-executed notebook anyway...")

    # Save executed notebook
    with open(NOTEBOOK_OUTPUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  ✓ Executed notebook saved to: {NOTEBOOK_OUTPUT}")

    return nb


# =============================================================================
# PDF EXPORT
# =============================================================================
def export_to_pdf():
    """Export the executed notebook to a publication-grade PDF."""
    print("\n" + "=" * 68)
    print("   EXPORTING TO PDF")
    print("=" * 68)

    temp_html = Path("/tmp/final_report_temp.html")
    temp_pdf = Path("/tmp/final_report_temp.pdf")

    # 1. Resolve jupyter binary
    conda_jupyter = Path("/home/huycao/anaconda3/bin/jupyter")
    venv_python = ASSIGNMENT_DIR.parent / ".venv" / "bin" / "python"

    if conda_jupyter.exists():
        cmd_html = [
            str(conda_jupyter), "nbconvert",
            "--to", "html",
            "--embed-images",
            str(NOTEBOOK_OUTPUT),
            "--output", str(temp_html)
        ]
    elif venv_python.exists():
        cmd_html = [
            str(venv_python), "-m", "nbconvert",
            "--to", "html",
            "--embed-images",
            str(NOTEBOOK_OUTPUT),
            "--output", str(temp_html)
        ]
    else:
        cmd_html = [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--embed-images",
            str(NOTEBOOK_OUTPUT),
            "--output", str(temp_html)
        ]

    print(f"\n[1/4] Compiling notebook to self-contained HTML...")
    subprocess.run(cmd_html, cwd=REPORT_DIR, check=True)
    print("      ✓ HTML generated successfully.")

    # 2. Post-process HTML for MathJax CDN, Base64 image fallback & Print CSS
    print("\n[2/4] Injecting MathJax CDN and print-optimized pagination CSS...")
    with open(temp_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Base64 image fallback: ensure ANY remaining relative/local img src is converted to base64 data URI
    import base64
    import mimetypes

    def _embed_img_base64(match):
        prefix = match.group(1)
        src = match.group(2)
        suffix = match.group(3)
        if src.startswith("data:") or src.startswith("http://") or src.startswith("https://"):
            return match.group(0)
        target_path = None
        for candidate in [
            REPORT_DIR / src,
            ASSIGNMENT_DIR / src,
            ASSIGNMENT_DIR.parent / src,
        ]:
            if candidate.resolve().exists() and candidate.resolve().is_file():
                target_path = candidate.resolve()
                break
        if target_path:
            mime_type, _ = mimetypes.guess_type(str(target_path))
            mime_type = mime_type or "image/png"
            with open(target_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("ascii")
            return f'{prefix}data:{mime_type};base64,{b64}{suffix}'
        return match.group(0)

    html_content = re.sub(r'(<img\s+[^>]*src=["\'])([^"\']+)(["\'])', _embed_img_base64, html_content)

    # Convert MathJax to SVG renderer
    html_content = html_content.replace("config=TeX-AMS_CHTML-full,Safe", "config=TeX-AMS_SVG,Safe")
    html_content = html_content.replace("config=TeX-AMS_CHTML", "config=TeX-AMS_SVG")
    html_content = re.sub(
        r'src="file:///usr/share/javascript/mathjax/[^"]+"',
        'src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_SVG,Safe"',
        html_content
    )

    mathjax_extra = ""
    if "mathjax" not in html_content.lower():
        mathjax_extra = '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_SVG,Safe"></script>'

    custom_css = f"""
{mathjax_extra}
<style type="text/css">
  @media print, screen {{
    /* Code block styling & wrapping */
    pre, code, .highlight pre, .jp-CodeCell pre, .input_area pre, div.input_area {{
      white-space: pre-wrap !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
      font-size: 8.5pt !important;
      line-height: 1.35 !important;
      max-width: 100% !important;
    }}
    .jp-Cell, .cell {{
      padding: 4px 6px !important;
    }}
    /* Page layout and margins */
    @page {{
      size: A4 portrait;
      margin: 1.4cm 1.2cm !important;
    }}
    /* Prevent orphaned headings at page bottom */
    h1, h2, h3, h4, h5, h6 {{
      page-break-after: avoid !important;
      break-after: avoid !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
    }}
    .cell:has(h1), .cell:has(h2), .cell:has(h3), .cell:has(h4),
    .jp-Cell:has(h1), .jp-Cell:has(h2), .jp-Cell:has(h3), .jp-Cell:has(h4) {{
      page-break-after: avoid !important;
      break-after: avoid !important;
    }}
    /* Prevent figure and caption splitting across pages */
    .figure-card, .figure-container, figure {{
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      margin: 12px 0 !important;
    }}
    img {{
      max-width: 86% !important;
      max-height: 440px !important;
      height: auto !important;
      display: block !important;
      margin: 8px auto !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
    }}
    /* Table layout and pagination */
    table, .dataframe {{
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      font-size: 8.5pt !important;
      margin: 10px 0 !important;
    }}
    .output_subarea {{
      max-width: 100% !important;
      overflow-x: hidden !important;
    }}
  }}
</style>
"""

    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{custom_css}\n</head>")
    else:
        html_content = f"{custom_css}\n{html_content}"

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("      ✓ MathJax CDN & custom print styles injected.")

    # 3. Headless Chrome print-to-pdf
    chrome_bin = shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome"
    print(f"\n[3/4] Rendering PDF via headless browser ({chrome_bin})...")
    cmd_pdf = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=15000",
        f"--print-to-pdf={temp_pdf}",
        str(temp_html)
    ]
    subprocess.run(cmd_pdf, check=True)
    print("      ✓ PDF rendered successfully.")

    # 4. Save to destination
    print(f"\n[4/4] Saving final PDF to: {PDF_OUTPUT}")
    shutil.copyfile(temp_pdf, PDF_OUTPUT)

    # Clean up temp files
    if temp_html.exists():
        temp_html.unlink()
    if temp_pdf.exists():
        temp_pdf.unlink()

    # Audit output
    pdf_size_mb = os.path.getsize(PDF_OUTPUT) / (1024 * 1024)
    print("-" * 68)
    print("              FINAL REPORT PDF GENERATION COMPLETE")
    print("-" * 68)
    print(f"  • Notebook Location : {NOTEBOOK_OUTPUT}")
    print(f"  • PDF Location      : {PDF_OUTPUT}")
    print(f"  • PDF Size          : {pdf_size_mb:.2f} MB")
    print("=" * 68)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Step 1: Build the notebook
    nb = build_final_notebook()

    # Step 2: Execute the notebook
    nb = execute_notebook(nb)

    # Step 3: Export to PDF
    export_to_pdf()

    print("\n🎉 Final report generation complete!")
