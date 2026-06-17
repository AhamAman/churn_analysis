# 📊 Telco Customer Churn Prediction & Explainability Dashboard

Customer churn is a critical problem for subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project develops an end-to-end Machine Learning system to predict customer churn probability and explain the primary drivers behind individual customer risks.

---

## 📌 1. The Data Problem

We use the standard **Telco Customer Churn Dataset** to predict whether a customer will leave (`Churn` = Yes/No) based on:
* **Demographics**: Gender, Senior Citizen status, Partner, Dependents.
* **Services**: Phone service, Multiple lines, Internet service type, Online security, Backup, Device protection, Tech support, Streaming TV/Movies.
* **Account Info**: Tenure (months), Contract type, Paperless billing, Payment method, Monthly charges, and Total charges.

---

## ⚙️ 2. Preprocessing & Design Decisions

### Feature Engineering & Preprocessing Pipeline
To prevent data leakage and ensure robust production inference, the preprocessing is bundled into a scikit-learn `ColumnTransformer` inside a unified `Pipeline`:
* **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`): Imputed with the median value and scaled using `StandardScaler`.
* **Categorical Features** (all service and billing columns): Imputed with the most frequent value and encoded using `OneHotEncoder(drop='if_binary')` or standard `OneHotEncoder` based on cardinalities.

### Class Imbalance Handling
The dataset is highly imbalanced (~26% churned, ~74% retained). Simply predicting the majority class would yield high accuracy but zero recall for churned customers.
* **Logistic Regression**: Trained using `class_weight='balanced'` to assign higher penalties to minority class errors.
* **Random Forest / XGBoost**: Tuned using weighted class criteria (`scale_pos_weight` in XGBoost) and setting optimized risk thresholds (e.g. classification threshold optimized at `0.3` or `0.6` instead of the default `0.5`).

---

## 📈 3. Model Training & Comparison

We trained and evaluated three distinct classifiers. Below is the performance summary:

| Model | ROC AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.84** | 0.62 | 0.50 | 0.80 |
| **Random Forest** | **0.84** | **0.63** | **0.54** | 0.75 |
| **XGBoost** (Tuned) | **0.84** | 0.62 | 0.51 | **0.81** |

* **Key Takeaway**: With hyperparameter tuning, **XGBoost** now achieves the highest **Recall (81%)**, capturing the highest percentage of at-risk customers, while **Random Forest** provides the highest overall **F1-Score (63%)** and **Precision (54%)**, making it the most balanced model. All models have identical raw ranking capabilities (**ROC AUC of 0.84**).

### 📝 Note: Why does Random Forest still yield a slightly higher Precision and F1-Score than XGBoost after tuning?
Even after running a hyperparameter search for both models, Random Forest retains a slight edge in **Precision (54% vs 51%)** and **F1-Score (63% vs 62%)**, while XGBoost leads in **Recall (81% vs 75%)**:
1. **Gradient Scaling Objective**: XGBoost uses `scale_pos_weight` (~3.0) to heavily penalize errors on the minority class (churners) during boosting. This aggressively pushes the model to prioritize Recall (capturing 81% of true churners) at the cost of a higher false-positive rate, which lowers its Precision to 51%. Random Forest balances class weights more conservatively.
2. **Variance Reduction via Bagging**: Random Forest trains trees independently using random subsets of data and features (bagging). This averages out noise, making it highly robust to overfitting on moderate tabular datasets (~7,000 rows). XGBoost builds trees sequentially to correct residual errors, which can be more sensitive to tabular noise unless regularized heavily.
3. **Identical Discriminative Power**: Both models share a **ROC AUC of 0.84**, indicating they rank customer churn risk with the exact same accuracy. The minor differences in Precision, Recall, and F1-Score are due to probability threshold calibration rather than one model being fundamentally superior.

---

## 🔍 4. Decision & Explainability Insights (SHAP)

Using **SHAP (SHapley Additive exPlanations)** on our ensemble models, we identified the top drivers influencing customer churn risk:

1. **Customer Tenure**: Shorter tenure is the single largest risk factor. New customers are highly susceptible to leaving.
2. **Contract Type**: Month-to-month contracts strongly correlate with high churn rates. Multi-year agreements act as strong retention anchors.
3. **Monthly Charges**: Higher monthly costs increase churn likelihood, suggesting price sensitivity.
4. **Value-Added Services**: A lack of Online Security, Tech Support, or Device Protection increases the probability of churn.

---

## 🚀 5. Getting Started & Running the App

The project includes a Streamlit web application providing a real-time churn prediction form and model feature impact plots.

### Prerequisites
Make sure you have Python installed. The project runs inside an isolated virtual environment.

### Setup Instructions
1. Clone or navigate to the project directory.
2. **Activate the virtual environment**:
   * **PowerShell**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Command Prompt (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
3. **Install Dependencies** (if setup from scratch):
   ```bash
   pip install -r requirements.txt
   ```

### Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the dashboard.