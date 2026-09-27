# Vietnam House Price Prediction - Intelligent System & Application

This project implements a complete intelligent system to predict residential house prices in Vietnam using a Random Forest Regressor. It includes data preprocessing, target encoding for high-cardinality location attributes, standard scaling, model training, controlled experiments, and deployment via a mobile-responsive web application.

The system realizes the complete pipeline:
`User Input → Representation → Preprocessing → ML Model → Price Prediction → Output`

---

## 1. Directory Structure

```text
Pờ-Rô-Jếct/
│
├── dataset/
│   └── vn_house.dataset.csv          # Raw housing listing dataset (30,229 rows)
│
├── model/
│   ├── base.ipynb                    # Main Jupyter notebook containing EDA, training, and analysis
│   ├── export_model.py               # Script to train, validate, and serialize model artifacts
│   ├── te.joblib                     # Serialized location Target Encoder
│   ├── rf_pipeline.joblib            # Serialized prediction pipeline (StandardScaler + RandomForest)
│   ├── preprocessing_metadata.joblib # Serialized column headers, continuous feature medians, schemas
│   ├── knn_hyperparameter_tuning.png # Hyperparameter tuning search visualization
│   └── model_comparison.png          # Model comparison test metrics visualization
│
├── static/
│   ├── index.html                    # Frontend user interface markup (Modern Flat UI layout)
│   ├── index.css                     # Mobile-responsive frontend styling
│   └── index.js                      # AJAX client handler & pipeline animation visualizer
│
├── app.py                            # Flask server hosting the predicting JSON API and static files
└── Task.md                           # Project status tracking and task checklist
```

---

## 2. Getting Started & Installation

To run the application locally on your computer or mobile simulator, follow these setup steps:

### Prerequisites
- Python 3.12 (or any Python 3.9+ version)
- `pip` package manager

### A. Environment Setup
1. Open your terminal and navigate to the project directory

2. Create a virtual environment named `.venv`:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Linux / macOS**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

4. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib flask
   ```

---

## 3. Running the System

The system can be run in two main steps:

### Step 1: Export/Train the Model (Optional)
If you wish to retrain the Random Forest model and re-generate the serialized files, run the export script:
```bash
python model/export_model.py
```
*Note: If the virtual environment is not activated, you can invoke the interpreter directly:*
```bash
.venv/bin/python model/export_model.py
```
This script will partition the dataset (80/20 train-test split), fit target encoding on locations, impute numerical features, run the random forest pipeline, validate metrics against the test set, and save the `.joblib` files to the `model` folder.

### Step 2: Launch the Web Application
Start the Flask backend web server:
```bash
python app.py
```
*Or, without activating the virtual environment:*
```bash
.venv/bin/python app.py
```

The server will start on port `5000` at:
👉 **`http://localhost:5000`**

Open this URL in a desktop web browser, or on a mobile phone web browser (or mobile simulator tool) connected to the same local network.

---

### Pipeline Flows Visualized:
When you click a demo button or calculate custom property values, the page visualizes each state:
- **Input**: Displaying raw user parameters.
- **Representation**: Extracting parsed categories (e.g. ward, district, city).
- **Preprocessing**: Visualizing standard scaled inputs and target encoded location predictions.
- **Model**: Outputting the Random Forest regression result (log scale).
- **Output**: Exponentiating and formatting the final VND amount.
# MonThayQue
