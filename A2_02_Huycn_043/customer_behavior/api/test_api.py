"""
Automated Verification Suite for Application 3 REST API & Web Server.
Tests endpoints, schemas, multimodal inference, and edge validation cases.
"""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from app import app, load_artifacts

# Ensure artifacts are loaded
load_artifacts()
client = TestClient(app)


def test_health():
    print("Testing GET /health ...")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("  [PASSED] /health returned healthy and model_loaded=True")


def test_model_info():
    print("Testing GET /model-info ...")
    response = client.get("/model-info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "champion_model" in data
    assert "target_classes" in data
    assert len(data["target_classes"]) == 5
    assert "metrics" in data
    print(f"  [PASSED] /model-info: Champion='{data['champion_model']}', Classes={data['target_classes']}")


def test_predictions():
    print("Testing POST /predict across diverse customer personas ...")
    cases = [
        {
            "name": "Dresses Enthusiast",
            "expected": "Dresses",
            "payload": {
                "Age": 32,
                "Rating": 5,
                "Recommended IND": 1,
                "Positive Feedback Count": 4,
                "Title": "Stunning summer maxi dress",
                "Review Text": "This dress fits like a glove! Beautiful floral fabric and perfect length for weddings."
            }
        },
        {
            "name": "Denim & Bottoms Shopper",
            "expected": "Bottoms",
            "payload": {
                "Age": 28,
                "Rating": 4,
                "Recommended IND": 1,
                "Positive Feedback Count": 0,
                "Title": "Great stretch denim jeans",
                "Review Text": "Love these high-waisted pants! Perfect fit around hips and ankles, durable denim material."
            }
        },
        {
            "name": "Winter Outerwear",
            "expected": "Jackets",
            "payload": {
                "Age": 45,
                "Rating": 5,
                "Recommended IND": 1,
                "Positive Feedback Count": 2,
                "Title": "Warm winter wool coat",
                "Review Text": "This tailored coat keeps me cozy and looks chic with boots. Heavy-weight lining and structured collar."
            }
        },
        {
            "name": "Intimates & Sleepwear",
            "expected": "Intimate",
            "payload": {
                "Age": 35,
                "Rating": 5,
                "Recommended IND": 1,
                "Positive Feedback Count": 1,
                "Title": "Soft silk lace bralette",
                "Review Text": "Extremely comfortable bra and underwear set. Delicate lace and great fit under shirts."
            }
        },
        {
            "name": "Casual Summer Tops",
            "expected": "Tops",
            "payload": {
                "Age": 50,
                "Rating": 4,
                "Recommended IND": 1,
                "Positive Feedback Count": 3,
                "Title": "Casual linen button-down blouse",
                "Review Text": "Great summer shirt to pair with shorts. Lightweight and breathable fabric with loose fit."
            }
        }
    ]

    for c in cases:
        resp = client.post("/predict", json=c["payload"])
        assert resp.status_code == 200, f"Failed on {c['name']}: {resp.text}"
        res = resp.json()
        pred = res["predicted_department"]
        conf = res["confidence"]
        assert pred == c["expected"], f"Expected {c['expected']}, got {pred} ({conf*100:.1f}%)"
        assert len(res["ranked_departments"]) == 5
        assert "word_count" in res["engineered_features"]
        assert res["engineered_features"]["word_count"] > 0
        print(f"  [PASSED] {c['name']} -> Predicted '{pred}' (Confidence: {conf*100:.2f}%), Behavior: '{res['behavior_category']}'")


def test_validation_errors():
    print("Testing input validation edge cases ...")
    # Empty review text
    resp = client.post("/predict", json={
        "Age": 30,
        "Rating": 5,
        "Recommended IND": 1,
        "Positive Feedback Count": 0,
        "Title": "Test",
        "Review Text": ""
    })
    assert resp.status_code == 422, f"Expected 422 for empty review text, got {resp.status_code}"
    print("  [PASSED] 422 received for empty review text")

    # Invalid Rating (> 5)
    resp = client.post("/predict", json={
        "Age": 30,
        "Rating": 10,
        "Recommended IND": 1,
        "Positive Feedback Count": 0,
        "Title": "Test",
        "Review Text": "Great shirt"
    })
    assert resp.status_code == 422, f"Expected 422 for rating=10, got {resp.status_code}"
    print("  [PASSED] 422 received for out-of-range rating")

    # Negative Age (< 18)
    resp = client.post("/predict", json={
        "Age": 12,
        "Rating": 4,
        "Recommended IND": 1,
        "Positive Feedback Count": 0,
        "Title": "Test",
        "Review Text": "Nice top"
    })
    assert resp.status_code == 422, f"Expected 422 for age < 18, got {resp.status_code}"
    print("  [PASSED] 422 received for underage input")


def test_frontend_routes():
    print("Testing frontend static routes ...")
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200 for index, got {resp.status_code}"
    assert "RetailSense AI" in resp.text
    print("  [PASSED] GET / returned index.html containing RetailSense AI")

    resp_css = client.get("/static/style.css")
    assert resp_css.status_code == 200
    print("  [PASSED] GET /static/style.css returned 200")

    resp_js = client.get("/static/app.js")
    assert resp_js.status_code == 200
    print("  [PASSED] GET /static/app.js returned 200")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING CUSTOMER BEHAVIOR API VERIFICATION SUITE")
    print("==================================================")
    test_health()
    test_model_info()
    test_predictions()
    test_validation_errors()
    test_frontend_routes()
    print("==================================================")
    print("ALL 5 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
