document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const btnSubmit = document.getElementById("btn-submit");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultDisplay = document.getElementById("result-display");
    const resultCard = document.getElementById("result-card");
    
    const priceBillion = document.getElementById("price-billion");
    const priceFormatted = document.getElementById("price-formatted");
    const predLog = document.getElementById("pred-log");

    // Form submission listener
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        
        // Parse inputs from form
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            if (["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"].includes(key)) {
                data[key] = value !== "" ? parseFloat(value) : null;
            } else {
                data[key] = value;
            }
        });

        // Set Address to empty string as Flask backend relies primarily on City_Province and District fallback
        data["Address"] = "";

        calculateValuation(data);
    });

    // Formats integer values as currency VND
    function formatCurrencyVND(number) {
        return new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' }).format(number);
    }

    // Call API and handle outputs
    function calculateValuation(payload) {
        // UI Loading States
        btnSubmit.disabled = true;
        btnSubmit.innerText = "Valuating...";
        resultPlaceholder.classList.remove("hidden");
        resultDisplay.classList.add("hidden");
        resultCard.classList.remove("success");
        resultPlaceholder.innerText = "Running Random Forest model calculations...";

        // Send AJAX Request
        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Valuation calculation failed on server.");
            }
            return response.json();
        })
        .then(res => {
            // Update valuation display card
            resultPlaceholder.classList.add("hidden");
            resultDisplay.classList.remove("hidden");
            resultCard.classList.add("success");

            priceBillion.innerText = `${res.price_billion_vnd.toFixed(3)} Billion VND`;
            priceFormatted.innerText = formatCurrencyVND(res.price_vnd);
            predLog.innerText = res.predicted_log.toFixed(6);
        })
        .catch(err => {
            resultPlaceholder.innerText = `Error: ${err.message}`;
            resultCard.classList.remove("success");
            console.error(err);
        })
        .finally(() => {
            btnSubmit.disabled = false;
            btnSubmit.innerText = "Valuate House";
        });
    }
});
