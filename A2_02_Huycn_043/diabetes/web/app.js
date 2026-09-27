/**
 * GlucoPredict AI — Frontend Controller
 * Handles user input submission, API communication, and dynamic results presentation.
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm");
  const submitBtn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const btnText = submitBtn.querySelector(".btn-text");

  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  const modelBadge = document.getElementById("modelBadge");

  const emptyState = document.getElementById("emptyState");
  const resultContent = document.getElementById("resultContent");
  const outcomeBanner = document.getElementById("outcomeBanner");
  const outcomeLabel = document.getElementById("outcomeLabel");
  const riskLevelBadge = document.getElementById("riskLevelBadge");

  const probabilityValue = document.getElementById("probabilityValue");
  const probabilityBar = document.getElementById("probabilityBar");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceBar = document.getElementById("confidenceBar");

  const interpretationText = document.getElementById("interpretationText");
  const glucoseBmiVal = document.getElementById("glucoseBmiVal");
  const insulinGlucoseVal = document.getElementById("insulinGlucoseVal");

  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");

  // Determine base API URL (relative or fallback for local static test)
  const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
    ? window.location.origin
    : "http://localhost:8000";

  // Check API health and retrieve model metadata on load
  async function checkApiHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === "healthy") {
          statusIndicator.className = "status-indicator online";
          statusText.textContent = "API Connected";
        }
      } else {
        throw new Error("API returned non-200");
      }

      // Fetch model info
      const infoRes = await fetch(`${API_BASE}/model-info`);
      if (infoRes.ok) {
        const info = await infoRes.json();
        if (info.champion_model) {
          modelBadge.textContent = `${info.champion_model} Champion`;
        }
      }
    } catch (err) {
      statusIndicator.className = "status-indicator offline";
      statusText.textContent = "API Disconnected";
    }
  }

  checkApiHealth();

  // Handle form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    // Collect and parse form input values
    const payload = {
      Pregnancies: parseInt(document.getElementById("pregnancies").value, 10),
      Glucose: parseFloat(document.getElementById("glucose").value),
      BloodPressure: parseFloat(document.getElementById("bloodPressure").value),
      SkinThickness: parseFloat(document.getElementById("skinThickness").value),
      Insulin: parseFloat(document.getElementById("insulin").value),
      BMI: parseFloat(document.getElementById("bmi").value),
      DiabetesPedigreeFunction: parseFloat(document.getElementById("dpf").value),
      Age: parseInt(document.getElementById("age").value, 10),
    };

    // Client-side quick check
    for (const [key, val] of Object.entries(payload)) {
      if (isNaN(val) || val < 0) {
        showError(`Please enter a valid non-negative number for ${key}.`);
        return;
      }
    }

    // Set loading state
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errDetail = await response.json().catch(() => ({ detail: "Inference server error" }));
        throw new Error(errDetail.detail || `Server returned code ${response.status}`);
      }

      const result = await response.json();
      renderPrediction(result);
    } catch (err) {
      showError(err.message || "Failed to communicate with prediction service.");
    } finally {
      setLoading(false);
    }
  });

  function renderPrediction(data) {
    // Hide empty placeholder, show active results
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    // Diagnosis label and color styling
    const isDiabetic = data.prediction === 1;
    outcomeLabel.textContent = data.label;

    if (isDiabetic) {
      outcomeBanner.className = "outcome-banner positive";
    } else {
      outcomeBanner.className = "outcome-banner negative";
    }

    // Risk badge
    riskLevelBadge.textContent = data.risk_level;
    if (data.risk_level === "High Risk") {
      riskLevelBadge.className = "risk-badge risk-high";
    } else if (data.risk_level === "Moderate Risk") {
      riskLevelBadge.className = "risk-badge risk-moderate";
    } else {
      riskLevelBadge.className = "risk-badge risk-low";
    }

    // Metrics & Bars
    const probPct = (data.probability * 100).toFixed(1);
    const confPct = (data.confidence * 100).toFixed(1);

    probabilityValue.textContent = `${probPct}%`;
    probabilityBar.style.width = `${probPct}%`;

    confidenceValue.textContent = `${confPct}%`;
    confidenceBar.style.width = `${confPct}%`;

    // Clinical interpretation
    interpretationText.textContent = data.interpretation;

    // Engineered features
    if (data.engineered_features) {
      glucoseBmiVal.textContent = data.engineered_features.Glucose_BMI_Interaction ?? "N/A";
      insulinGlucoseVal.textContent = data.engineered_features.Insulin_Glucose_Ratio ?? "N/A";
    }

    // Smooth scroll to result on mobile
    if (window.innerWidth <= 860) {
      document.getElementById("resultSection").scrollIntoView({ behavior: "smooth" });
    }
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
      spinner.classList.remove("hidden");
      btnText.textContent = "Assessing Risk...";
    } else {
      spinner.classList.add("hidden");
      btnText.textContent = "Execute Clinical Assessment";
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove("hidden");
  }

  function hideError() {
    errorAlert.classList.add("hidden");
    errorMessage.textContent = "";
  }
});
