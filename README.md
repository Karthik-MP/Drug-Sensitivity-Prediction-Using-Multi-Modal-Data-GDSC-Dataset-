# Drug Sensitivity Prediction Using Multi-Modal Data (GDSC Dataset)

**Course**: CS581A/481A - Multi-modal Machine Learning in Biomedicine

## 📌 Project Description

This project aims to predict drug response (**measured as LN_IC50**) for cancer cell lines using **multi-modal biological data** from the **Genomics of Drug Sensitivity in Cancer (GDSC)** dataset.

By integrating **genomic** and **transcriptomic** features, we developed a machine learning pipeline to predict how sensitive different cell lines are to various anti-cancer drugs.

---

## 🔁 Machine Learning Workflow

### 1. Preprocessing
- **Drug-wise missing value imputation** using:
  - Statistical methods
  - K-Nearest Neighbors (KNN)
  - Random Forest imputation
- **Feature encoding** strategies:
  - Binary Encoding
  - One-Hot Encoding
  - Target Encoding
  - Label Encoding

### 2. Model
- **XGBoost Regressor** trained on 80% of the dataset.
- **Hyperparameter tuning** using `RandomizedSearchCV`.

### 3. Evaluation Metrics
- **R² Score**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

### 4. Interpretability
- Model explanations generated using **SHAP (SHapley Additive Explanations)** to assess feature importance and influence on predictions.

---

## 📂 File Structure

```plaintext
.
├── gdsc_xgboost_model.pkl             # Trained XGBoost model (saved with joblib)
├── Gdsc Project Presentation.pptx     # Final presentation slides
├── STATEMENT.txt                      # Academic honesty statement
├── README.txt                         # Instructions to run the code
├── notebooks/                         # Jupyter notebooks (preprocessing, modeling, SHAP)
│   └── gdsc_prediction_pipeline.ipynb
├── data/                              # Raw and cleaned dataset files

## Open the Jupyter Notebook:
- notebooks/gdsc_prediction_pipeline.ipynb
- pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost shap

