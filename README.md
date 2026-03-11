# 💰 Lanka Micro-Finance AI
### Alternative Credit Scoring for Sri Lankan Micro-Entrepreneurs

> Bridging the financial inclusion gap through behavioral data science

---

## 1. Project Vision and Problem Statement

In Sri Lanka, millions of street vendors, home-based workers, and freelancers are invisible to traditional banking systems. Not because they are untrustworthy, but simply because they do not have a formal CRIB credit record to present to a bank manager.

This project was built around one clear mission: to evaluate **creditworthiness** instead of just **credit history**. By reading behavioral signals from everyday digital and financial activity, this AI system gives micro-entrepreneurs a fair shot at accessing the loans they need to grow their businesses.

---

## 2. The Data Strategy

Traditional scoring models look at income statements and bank records. This system looks at **behavioral consistency** instead.

The primary dataset (`lanka_microfinance_data.csv`, 1,000 records) was purpose-built for this prototype and captures five categories of alternative signals:

| Signal Category | Feature | Why It Matters |
|---|---|---|
| Financial Hygiene | `Utility_Bill_Late_Days` | Late electricity or water payments reveal cash flow stress |
| Cash Flow Proxy | `Mobile_Reload_Consistency` | Regular mobile top-ups signal steady disposable income |
| Digital Adoption | `Digital_Literacy_Score` | Interaction speed and accuracy reflect financial engagement habits |
| Social Capital | `Community_Group_Member` | Membership in Samurdhi, trade groups, or death-donation societies |
| Anchor Variables | `Age`, `Monthly_Income_LKR`, `Existing_Loans`, `Dependents` | Traditional grounding features for context |

### What the Primary Dataset (V1) Tells Us

| Observation | Value |
|---|---|
| Overall default rate | 24.2% |
| Default rate with 0 existing loans | 0.9% |
| Default rate with 1 existing loan | 19.2% |
| Default rate with 2 existing loans | 55.4% |
| Default rate among community members | 23.2% |
| Default rate among non-members | 25.6% |
| Correlation of `Existing_Loans` with default | +0.51 (strongest predictor) |
| Correlation of `Utility_Bill_Late_Days` with default | +0.36 |
| Correlation of `Mobile_Reload_Consistency` with default | -0.39 |

The numbers tell a clear story. An applicant carrying two or more active loans is already in a structurally high-risk position, regardless of income level or business type. Mobile reload consistency shows a strong inverse relationship with default, confirming it as one of the most reliable behavioral proxies in this context.

---

## 3. EDA Highlights

Three visualisations were produced during exploratory analysis to understand the distribution of the target variable and the separating power of the two strongest behavioral features.

**Class Distribution**

![Loan Default vs Non-Default Count](https://github.com/SLxnoat/Lanka-Microfinance-AI/blob/main/images/Loan%20Default%20vs%20Non-Default%20Count.png)

The dataset holds a 75.8% to 24.2% split between non-default and default cases, reflecting a realistic loan portfolio composition for a Sri Lankan micro-finance context.

**Mobile Reload Consistency vs Loan Default**

![Mobile Reload Consistency vs Loan Default](https://github.com/SLxnoat/Lanka-Microfinance-AI/blob/main/images/Mobile%20Reload%20Consistency%20vs%20Loan%20Default.png)

Non-defaulters cluster clearly higher, with a median around 0.68, compared to 0.44 for defaulters. The overlap between distributions is intentional — this feature contributes meaningful signal without being a standalone predictor, which is the expected behavior of a genuine behavioral proxy.

---

## 4. Model Engineering

The core prediction engine is an **XGBoost Classifier**, chosen for its strong performance on tabular data and its ability to capture non-linear relationships between features.

### Training Configuration

The model was trained using an 80/20 train-test split with the following hyperparameters:

```python
XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=3
)
```

The `scale_pos_weight=3` parameter directly addresses class imbalance by applying a cost-sensitive weight to the minority class, calculated as:

$$scale\_pos\_weight = \frac{\text{Total Negative Samples}}{\text{Total Positive Samples}} \approx 3.0$$

### Feature Importance by Gain

| Rank | Feature | Gain Score |
|---|---|---|
| 1 | `Existing_Loans` | 19.55 |
| 2 | `Mobile_Reload_Consistency` | 7.27 |
| 3 | `Utility_Bill_Late_Days` | 5.73 |
| 4 | `Monthly_Income_LKR` | 0.83 |
| 5 | `Business_Type` | 0.53 |
| 6 | `Dependents` | 0.51 |
| 7 | `Age` | 0.40 |
| 8 | `Digital_Literacy_Score` | 0.24 |

### Test Set Performance (V1 Dataset, 200 samples)

```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       161
           1       0.90      0.90      0.90        39

    accuracy                           0.96       200
```

```
Confusion Matrix:
[[157   4]
 [  4  35]]
```

The model achieves 96% accuracy with a balanced confusion matrix of 4 false negatives and 4 false positives on the test set.

---

## 5. Technical Reasoning and Trade-off Analysis

This section documents the key design decisions and debates that shaped the project, developed through iterative review across Claude and Gemini.

---

### 5.1 V1 as a Research Prototype: A Deliberate Starting Point

The primary dataset was built with a controlled, deterministic label structure:

```python
score = (Utility_Bill_Late_Days * 0.4) - (Mobile_Reload_Consistency * 15) + (Existing_Loans * 5)
Loan_Default = (score > 5)
```

This was a deliberate research choice. By anchoring the label to a known formula, the prototype was designed to first validate the full pipeline — data generation, model training, threshold analysis, and the Streamlit deployment — in a controlled, reproducible environment before introducing the complexity of real-world noise.

The V1 model's 96% accuracy and ROC-AUC of 0.9996 are therefore understood as **upper-bound prototype benchmarks**, not real-world performance claims.

---

### 5.2 V2 as a Realistic Extension

The second dataset (`lanka_microfinance_data_v2.csv`) was developed as a natural evolution of the prototype. It replaces the deterministic label formula with a multi-factor weighted risk score plus Gaussian noise, and introduces realistic inter-feature correlations. For example, income is now inversely correlated with late days, and mobile reload consistency is tied to income level.

The result is a more honest prediction environment, where the model achieves 87% accuracy on the test set.

| Metric | V1 Prototype | V2 Realistic Extension |
|---|---|---|
| Overall Accuracy | 96% | 87% |
| Precision on High Risk | 90% | 75% |
| Recall on Default Detection | 90% | 81% |
| False Negatives | 4 | 11 |
| False Positives | 4 | 15 |
| `Income` correlation with default | +0.008 (noise) | -0.075 (realistic) |
| `Community_Group_Member` correlation | -0.028 (weak) | -0.143 (meaningful) |

The drop from 96% to 87% is not a regression. It is the expected cost of replacing a clean formula with real-world behavioral complexity, and it is the direction this project is headed.

---

### 5.3 Threshold Strategy

XGBoost defaults to a classification threshold of 0.5. Lowering it makes the model stricter about flagging applicants as high risk:

| Threshold | False Negatives | False Positives | Business Impact |
|---|---|---|---|
| 0.3 | 4 | 7 | Over-rejects good applicants |
| **0.4** | **4** | **5** | Recommended — balanced risk and revenue |
| 0.5 | 4 | 4 | May miss borderline cases in noisier data |

A threshold of 0.3 minimises missed defaults but causes too many creditworthy applicants to be turned away. Since a micro-finance business generates its revenue from loan interest, over-rejection directly damages sustainability. A threshold of 0.4 was selected as the practical balance between protecting the lender and keeping loan disbursement at a viable level.

---

### 5.4 Repayment Intent vs Repayment Capacity

The `Community_Group_Member` feature was included under the assumption that members of Samurdhi, trade associations, or death-donation societies (මරණාධාර සමිති) face enough social pressure to honour their repayments. The V1 data shows a small directional gap: community members default at 23.2% versus 25.6% for non-members. The V2 extension widens this to 19.1% versus 31.5%, which is more meaningful.

However, two concepts that are easy to conflate here are repayment intent and repayment capacity. Intent refers to the motivation to repay in order to protect social standing. Capacity refers to whether the applicant actually has the disposable income to make repayments on time. Samurdhi beneficiaries are by definition a lower-income population, which means their financial capacity is often most constrained precisely where their social intent is strongest.

Future iterations should treat these as separate signals by introducing derived features such as `loan_to_income_ratio` to represent capacity explicitly, rather than relying on community membership as a combined proxy.

---

## 6. Tech Stack and Project Structure

| Layer | Technology |
|---|---|
| ML Engine | XGBoost |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Web Interface | Streamlit |
| Model Serialisation | Joblib |

```
lanka-microfinance-ai/
│
├── app.py                          # Streamlit prediction dashboard
├── model_training.ipynb            # ML pipeline — train, evaluate, save
├── eda_analysis.ipynb              # Exploratory data analysis
├── data_generator.ipynb            # Synthetic dataset generation
│
├── data/
│   ├── lanka_microfinance_data.csv       # V1 — primary prototype dataset
│   ├── lanka_microfinance_data_v2.csv    # V2 — realistic extension dataset
│   └── microfinance_model.pkl            # Serialised XGBoost model
│
└── requirements.txt
```

---

## 7. Setup and Installation

Clone the repository:

```bash
git clone https://github.com/SLxnoat/Lanka-Microfinance-AI.git
cd Lanka-Microfinance-AI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch the prediction dashboard:

```bash
streamlit run app.py
```

The `microfinance_model.pkl` file must be placed inside the `data/` folder before running the app.

---

## 8. Roadmap

-  Integrate SHAP values to explain individual loan decisions to applicants and field officers
-  Use precision-recall curves with Optuna to calibrate the classification threshold per lender risk profile
-  Retrain the model on the V2 dataset and evaluate cross-validated performance
-  Connect to telecom provider APIs for real-time mobile reload consistency data
-  Run a fairness audit across business types, age groups, and income brackets
-  Build derived interaction features such as `income_per_dependent` and `loan_to_income_ratio`
-  Develop a lightweight React Native field app for offline data collection by loan officers

---

## Author

**Mayura Bandara**  
IT Undergraduate · ML Enthusiast · Founder of ArtXpert Design

[LinkedIn](https://www.linkedin.com/in/charuka-mayura)

---

> **Disclaimer:** This is a research prototype. Financial institutions should conduct thorough stress testing, bias audits, and regulatory review before any production deployment.
