# Application Progression Rule: 5-Stage Lifecycle

Every application in Assignment 02 follows the 5-stage lifecycle progression:

$$\text{Raw Data} \longrightarrow \text{Understand \& Clean} \longrightarrow \text{Represent} \longrightarrow \text{Learn} \longrightarrow \text{Evaluate} \longrightarrow \text{Persist} \longrightarrow \text{Deploy (Web)}$$

1. **Stage 1: Data Understanding, Representation & EDA**
   - Data stored in `Assignment_02/<app>/data/`
   - Follow 23-section notebook structure (Appendix B)
   - Formalize representation: $x_i \to X \in \mathbb{R}^{N \times d}, y$
   - 3-point EDA explanations: Observation, Interpretation, ML Implication

2. **Stage 2: Preprocessing, Modeling & Evaluation**
   - Train/test split with zero leakage
   - Scikit-learn Pipeline with fitted preprocessors
   - Compare required candidate models (5 for Diabetes/Housing, 6 for E-commerce)
   - Complete metric evaluation & Confusion Matrix / Residual analysis

3. **Stage 3: Persistence & Offline Inference**
   - Save `preprocessor.joblib` and `model.joblib` under `Assignment_02/<app>/model/`
   - Test standalone inference

4. **Stage 4: Web Deployment (API & UI)**
   - REST API (`POST /predict`) with schema validation under `Assignment_02/<app>/api/`
   - Web frontend under `Assignment_02/<app>/web/`
   - Note: Mobile deployment is excluded per user specification

5. **Stage 5: Verification & Deliverables**
   - Capture Web UI screenshots for final technical report
   - Update `assignment_02_tasks.md`
