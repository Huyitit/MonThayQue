/**
 * HouseVal AI — Frontend Controller
 * Application 2: Residential House Price Prediction (Regression)
 * Handles preset switching, user input submission, API communication, and dynamic results presentation.
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("valuationForm");
  const submitBtn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const btnText = submitBtn.querySelector(".btn-text");

  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  const modelBadge = document.getElementById("modelBadge");

  const emptyState = document.getElementById("emptyState");
  const resultContent = document.getElementById("resultContent");

  const predictedPriceValue = document.getElementById("predictedPriceValue");
  const formattedVndValue = document.getElementById("formattedVndValue");
  const valuationStatusBadge = document.getElementById("valuationStatusBadge");

  const pricePerM2Value = document.getElementById("pricePerM2Value");
  const rangeIntervalValue = document.getElementById("rangeIntervalValue");
  const rangeNote = document.getElementById("rangeNote");

  const totalFloorAreaVal = document.getElementById("totalFloorAreaVal");
  const bedBathRatioVal = document.getElementById("bedBathRatioVal");
  const roomDensityVal = document.getElementById("roomDensityVal");
  const recapGrid = document.getElementById("recapGrid");

  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");
  const presetButtons = document.querySelectorAll(".btn-preset");

  // Determine base API URL
  const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
    ? window.location.origin
    : "http://localhost:8001";

  // Presets definition
  const PRESETS = {
    hcmc: {
      Area: 75.0,
      Frontage: 4.5,
      "Access Road": 6.0,
      Floors: 3.0,
      Bedrooms: 3.0,
      Bathrooms: 3.0,
      Province_City: "Hồ Chí Minh",
      "Legal status": "Have certificate",
    },
    hanoi: {
      Area: 120.0,
      Frontage: 6.0,
      "Access Road": 8.0,
      Floors: 2.0,
      Bedrooms: 4.0,
      Bathrooms: 3.0,
      Province_City: "Hà Nội",
      "Legal status": "Have certificate",
    },
    binhduong: {
      Area: 60.0,
      Frontage: 4.0,
      "Access Road": 4.0,
      Floors: 1.0,
      Bedrooms: 2.0,
      Bathrooms: 1.0,
      Province_City: "Bình Dương",
      "Legal status": "Have certificate",
    },
  };

  function applyPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    document.getElementById("area").value = data.Area;
    document.getElementById("frontage").value = data.Frontage;
    document.getElementById("accessRoad").value = data["Access Road"];
    document.getElementById("floors").value = data.Floors;
    document.getElementById("bedrooms").value = data.Bedrooms;
    document.getElementById("bathrooms").value = data.Bathrooms;
    document.getElementById("provinceCity").value = data.Province_City;
    document.getElementById("legalStatus").value = data["Legal status"];

    presetButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-preset") === presetKey);
    });
  }

  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      applyPreset(presetKey);
    });
  });

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
      Area: parseFloat(document.getElementById("area").value),
      Frontage: parseFloat(document.getElementById("frontage").value),
      "Access Road": parseFloat(document.getElementById("accessRoad").value),
      Floors: parseFloat(document.getElementById("floors").value),
      Bedrooms: parseFloat(document.getElementById("bedrooms").value),
      Bathrooms: parseFloat(document.getElementById("bathrooms").value),
      Province_City: document.getElementById("provinceCity").value,
      "Legal status": document.getElementById("legalStatus").value,
    };

    // Client-side validation
    if (isNaN(payload.Area) || payload.Area < 10) {
      showError("Please enter a valid property land area (>= 10 m²).");
      return;
    }
    if (isNaN(payload.Frontage) || payload.Frontage <= 0) {
      showError("Please enter a valid frontage width (> 0 m).");
      return;
    }
    if (isNaN(payload["Access Road"]) || payload["Access Road"] <= 0) {
      showError("Please enter a valid access road width (> 0 m).");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errDetail = await response.json().catch(() => ({ detail: "Inference server error" }));
        throw new Error(errDetail.detail || `Server returned status ${response.status}`);
      }

      const result = await response.json();
      renderPrediction(result, payload);
    } catch (err) {
      showError(err.message || "Failed to communicate with prediction service.");
    } finally {
      setLoading(false);
    }
  });

  function renderPrediction(data, inputPayload) {
    // Hide empty placeholder, show active results
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    // Primary Price display
    predictedPriceValue.textContent = `${data.predicted_price_billion.toFixed(3)} Billion VND`;
    formattedVndValue.textContent = `≈ ${data.formatted_price_vnd}`;

    // Metrics
    pricePerM2Value.textContent = `${data.price_per_m2_million.toFixed(2)} M/m²`;
    rangeIntervalValue.textContent = `${data.valuation_range.low_estimate_billion.toFixed(2)}B - ${data.valuation_range.high_estimate_billion.toFixed(2)}B`;
    rangeNote.textContent = data.valuation_range.confidence_interval_note;

    // Engineered Features
    if (data.engineered_features) {
      totalFloorAreaVal.textContent = `${data.engineered_features.Total_Floor_Area.toFixed(1)} m²`;
      bedBathRatioVal.textContent = data.engineered_features.Bed_Bath_Ratio.toFixed(2);
      roomDensityVal.textContent = data.engineered_features.Room_Density.toFixed(4);
    }

    // Listing Recap
    recapGrid.innerHTML = `
      <div class="recap-item">Location: <strong>${inputPayload.Province_City}</strong></div>
      <div class="recap-item">Area: <strong>${inputPayload.Area} m²</strong></div>
      <div class="recap-item">Frontage: <strong>${inputPayload.Frontage} m</strong></div>
      <div class="recap-item">Access Road: <strong>${inputPayload["Access Road"]} m</strong></div>
      <div class="recap-item">Floors: <strong>${inputPayload.Floors}</strong></div>
      <div class="recap-item">Beds / Baths: <strong>${inputPayload.Bedrooms} / ${inputPayload.Bathrooms}</strong></div>
      <div class="recap-item" style="grid-column: span 2;">Legal: <strong>${inputPayload["Legal status"]}</strong></div>
    `;

    // Smooth scroll to result on mobile
    if (window.innerWidth <= 900) {
      document.getElementById("resultSection").scrollIntoView({ behavior: "smooth" });
    }
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
      spinner.classList.remove("hidden");
      btnText.textContent = "Valuating Property...";
    } else {
      spinner.classList.add("hidden");
      btnText.textContent = "Estimate Property Value";
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
