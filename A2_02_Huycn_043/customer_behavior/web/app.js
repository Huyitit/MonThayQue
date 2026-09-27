/**
 * RetailSense AI — Frontend Controller
 * Application 3: E-Commerce Customer Behavior (Interest Discovery)
 * Handles customer persona presets, form validation, REST API inference, and dynamic results visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Form Controls
  const form = document.getElementById("predictionForm");
  const submitBtn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const btnText = submitBtn.querySelector(".btn-text");

  const ageInput = document.getElementById("age");
  const ratingSelect = document.getElementById("rating");
  const recSelect = document.getElementById("recommendedInd");
  const feedbackInput = document.getElementById("positiveFeedback");
  const titleInput = document.getElementById("reviewTitle");
  const reviewTextInput = document.getElementById("reviewText");
  const charCounter = document.getElementById("charCounter");

  // DOM Elements - Status & Badges
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  const modelBadge = document.getElementById("modelBadge");

  // DOM Elements - Result Section
  const emptyState = document.getElementById("emptyState");
  const resultContent = document.getElementById("resultContent");
  const behaviorBadge = document.getElementById("behaviorBadge");
  const deptIcon = document.getElementById("deptIcon");
  const predictedDept = document.getElementById("predictedDept");

  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceBar = document.getElementById("confidenceBar");
  const probBarsContainer = document.getElementById("probBarsContainer");
  const interpretationText = document.getElementById("interpretationText");

  // Feature Engineering Pills
  const wordCountVal = document.getElementById("wordCountVal");
  const reviewLengthVal = document.getElementById("reviewLengthVal");
  const logFeedbackVal = document.getElementById("logFeedbackVal");
  const upperRatioVal = document.getElementById("upperRatioVal");
  const hasTitleVal = document.getElementById("hasTitleVal");

  // Error Alert
  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");

  // Preset Buttons
  const presetButtons = document.querySelectorAll(".btn-preset");

  // Department Icon Map
  const DEPT_ICONS = {
    Dresses: "👗",
    Bottoms: "👖",
    Jackets: "🧥",
    Intimate: "👙",
    Tops: "👚",
  };

  // Presets Data Dictionary
  const PRESETS = {
    dresses: {
      Age: 32,
      Rating: "5",
      RecommendedIND: "1",
      PositiveFeedback: 4,
      Title: "Stunning summer maxi dress",
      ReviewText: "This dress fits like a glove! Beautiful floral fabric and perfect length for weddings.",
    },
    bottoms: {
      Age: 28,
      Rating: "4",
      RecommendedIND: "1",
      PositiveFeedback: 0,
      Title: "Great stretch denim jeans",
      ReviewText: "Love these high-waisted pants! Perfect fit around hips and ankles, durable denim material.",
    },
    jackets: {
      Age: 45,
      Rating: "5",
      RecommendedIND: "1",
      PositiveFeedback: 2,
      Title: "Warm winter wool coat",
      ReviewText: "This tailored coat keeps me cozy and looks chic with boots. Heavy-weight lining and structured collar.",
    },
    intimate: {
      Age: 35,
      Rating: "5",
      RecommendedIND: "1",
      PositiveFeedback: 1,
      Title: "Soft silk lace bralette",
      ReviewText: "Extremely comfortable bra and underwear set. Delicate lace and great fit under shirts.",
    },
    tops: {
      Age: 50,
      Rating: "4",
      RecommendedIND: "1",
      PositiveFeedback: 3,
      Title: "Casual linen button-down blouse",
      ReviewText: "Great summer shirt to pair with shorts. Lightweight and breathable fabric with loose fit.",
    },
  };

  // Determine base API URL
  const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
    ? window.location.origin
    : "http://127.0.0.1:8002";

  // Real-time Text Counter
  function updateTextCounter() {
    const text = reviewTextInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = text.length;
    charCounter.textContent = `${words} words · ${chars} chars`;
  }
  reviewTextInput.addEventListener("input", updateTextCounter);
  updateTextCounter();

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

      const infoRes = await fetch(`${API_BASE}/model-info`);
      if (infoRes.ok) {
        const info = await infoRes.json();
        if (info.champion_model) {
          modelBadge.textContent = "Multimodal LR Champion";
        }
      }
    } catch (err) {
      statusIndicator.className = "status-indicator offline";
      statusText.textContent = "API Disconnected";
    }
  }

  checkApiHealth();

  // Load Preset Handler
  function loadPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    ageInput.value = data.Age;
    ratingSelect.value = data.Rating;
    recSelect.value = data.RecommendedIND;
    feedbackInput.value = data.PositiveFeedback;
    titleInput.value = data.Title;
    reviewTextInput.value = data.ReviewText;

    updateTextCounter();
    hideError();

    // Trigger prediction automatically
    executeInference();
  }

  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const preset = btn.getAttribute("data-preset");
      loadPreset(preset);
    });
  });

  // UI Error Display Helpers
  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove("hidden");
  }

  function hideError() {
    errorAlert.classList.add("hidden");
    errorMessage.textContent = "";
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
      spinner.classList.remove("hidden");
      btnText.textContent = "Classifying Interest...";
    } else {
      spinner.classList.add("hidden");
      btnText.textContent = "Discover Product Interest";
    }
  }

  // Execute Inference Function
  async function executeInference() {
    hideError();

    const age = parseInt(ageInput.value, 10);
    const rating = parseInt(ratingSelect.value, 10);
    const recInd = parseInt(recSelect.value, 10);
    const feedback = parseInt(feedbackInput.value, 10);
    const title = titleInput.value.trim();
    const reviewText = reviewTextInput.value.trim();

    // Basic Validation
    if (isNaN(age) || age < 18 || age > 120) {
      showError("Please enter a valid age between 18 and 120.");
      return;
    }
    if (isNaN(feedback) || feedback < 0) {
      showError("Positive feedback count must be non-negative.");
      return;
    }
    if (!reviewText) {
      showError("Please provide customer review text to execute multimodal interest discovery.");
      return;
    }

    const payload = {
      Age: age,
      Rating: rating,
      "Recommended IND": recInd,
      "Positive Feedback Count": feedback,
      Title: title,
      "Review Text": reviewText,
    };

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        const detail = errorData.detail || "Prediction request failed.";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      showError(err.message || "Failed to connect to inference server.");
    } finally {
      setLoading(false);
    }
  }

  // Render Inference Results
  function renderResults(res) {
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    // Winning Department Banner
    const dept = res.predicted_department;
    predictedDept.textContent = dept;
    deptIcon.textContent = DEPT_ICONS[dept] || "🏷️";

    // Behavior Badge
    const category = res.behavior_category || "Standard Shopper";
    behaviorBadge.textContent = category;
    behaviorBadge.className = "risk-badge";
    if (category.includes("Promoter") || category.includes("Advocate")) {
      behaviorBadge.classList.add("promoter");
    } else if (category.includes("Critical") || category.includes("Risk")) {
      behaviorBadge.classList.add("critical");
    } else {
      behaviorBadge.classList.add("evaluator");
    }

    // Confidence Gauge
    const confPct = (res.confidence * 100).toFixed(1);
    confidenceValue.textContent = `${confPct}%`;
    confidenceBar.style.width = `${Math.min(confPct, 100)}%`;

    // Multi-Class Distribution Bars
    probBarsContainer.innerHTML = "";
    if (res.ranked_departments && res.ranked_departments.length > 0) {
      res.ranked_departments.forEach((item) => {
        const isTop = item.department === dept;
        const pct = item.percentage.toFixed(1);
        const icon = DEPT_ICONS[item.department] || "🏷️";

        const row = document.createElement("div");
        row.className = `prob-row ${isTop ? "top-class" : ""}`;
        row.innerHTML = `
          <div class="prob-row-header">
            <span class="prob-row-dept">${icon} ${item.department}</span>
            <span class="prob-row-val">${pct}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${Math.min(pct, 100)}%;"></div>
          </div>
        `;
        probBarsContainer.appendChild(row);
      });
    }

    // Personalized Retail Strategy
    interpretationText.textContent = res.interpretation || "No recommendation generated.";

    // Engineered Features
    if (res.engineered_features) {
      const eng = res.engineered_features;
      wordCountVal.textContent = eng.word_count ?? 0;
      reviewLengthVal.textContent = eng.review_length ?? 0;
      logFeedbackVal.textContent = (eng.log_positive_feedback ?? 0).toFixed(2);
      upperRatioVal.textContent = `${((eng.uppercase_ratio ?? 0) * 100).toFixed(1)}%`;
      hasTitleVal.textContent = eng.has_title === 1 ? "Yes" : "No";
    }
  }

  // Form Submit Handler
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    executeInference();
  });
});
