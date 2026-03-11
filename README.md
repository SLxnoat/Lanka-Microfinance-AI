# \# 💰 Lanka Micro-Finance AI

# \### Alternative Credit Scoring for Sri Lankan Micro-Entrepreneurs

# 

# > Bridging the financial inclusion gap through behavioral data science

# 

# ---

# 

# \## 🌟 1. Project Vision \& Problem Statement

# 

# In Sri Lanka, millions of street vendors, home-based workers, and freelancers are \*\*invisible to traditional banking systems\*\* not because they are untrustworthy, but because they lack a formal CRIB credit record.

# 

# \*\*The Mission:\*\* Build a data-driven credit risk engine that evaluates \*creditworthiness\* instead of just \*credit history\*. By reading behavioral signals from everyday digital and financial activity, this system empowers the unbanked to access the microloans they need.

# 

# ---

# 

# \## 🧬 2. The Data Strategy — Alternative Data Points

# 

# Traditional models look at income statements. We look at \*\*behavioral consistency\*\*.  

# Our dataset (`lanka\_microfinance\_data\_v2.csv`, 1,000 records) captures:

# 

# | Signal | Feature | Rationale |

# |---|---|---|

# | 💧 Financial Hygiene | `Utility\_Bill\_Late\_Days` | Late electricity/water payments reflect cash flow stress |

# | 📱 Cash Flow Proxy | `Mobile\_Reload\_Consistency` | Regular top-ups signal steady disposable income |

# | 🖥️ Digital Adoption | `Digital\_Literacy\_Score` | Interaction speed/accuracy as a proxy for financial engagement |

# | 🤝 Social Capital | `Community\_Group\_Member` | Membership in trade groups, Samurdhi, or death-donation societies |

# | 📊 Anchor Variables | `Age`, `Monthly\_Income\_LKR`, `Existing\_Loans`, `Dependents` | Grounding features for context |

# 

# \### Key Dataset Insights (V2)

# 

# | Observation | Value |

# |---|---|

# | Overall default rate | \*\*25.0%\*\* |

# | Default rate — 0 existing loans | 1.1% |

# | Default rate — 1 existing loan | 22.2% |

# | Default rate — 2 existing loans | \*\*72.2%\*\* |

# | Default rate — 3 existing loans | \*\*96.4%\*\* |

# | Default rate — community members | 19.1% |

# | Default rate — non-members | 31.5% |

# | `Existing\_Loans` correlation with default | +0.68 (strongest predictor) |

# | `Utility\_Bill\_Late\_Days` correlation | +0.41 |

# | `Mobile\_Reload\_Consistency` correlation | -0.18 |

# 

# ---

# 

# \## 🛠️ 3. Model Engineering \& Optimisation

# 

# \*\*Core algorithm:\*\* XGBoost Classifier — selected for its superior performance on tabular data and ability to model non-linear feature interactions.

# 

# \### Handling Class Imbalance

# 

# The dataset holds a realistic ~3:1 ratio of good to bad loans. To prevent the model from being biased toward the majority class, we applied \*\*cost-sensitive learning\*\* via the `scale\_pos\_weight` parameter:

# 

# $$scale\\\_pos\\\_weight = \\frac{\\text{Total Negative Samples}}{\\text{Total Positive Samples}} = \\frac{750}{250} = 3.0$$

# 

# \### Feature Importance (by Gain)

# 

# | Rank | Feature | Gain Score |

# |---|---|---|

# | 1 | `Existing\_Loans` | 19.55 |

# | 2 | `Mobile\_Reload\_Consistency` | 7.27 |

# | 3 | `Utility\_Bill\_Late\_Days` | 5.73 |

# | 4 | `Monthly\_Income\_LKR` | 0.83 |

# | 5 | `Business\_Type` | 0.53 |

# | 6 | `Dependents` | 0.51 |

# | 7 | `Age` | 0.40 |

# | 8 | `Digital\_Literacy\_Score` | 0.24 |

# 

# \### Performance Evolution

# 

# | Metric | Synthetic Baseline (V1) | Realistic Scenario (V2) |

# |---|---|---|

# | Overall Accuracy | 96% | \*\*87%\*\* |

# | Precision (High Risk) | 90% | 75% |

# | Recall (Default Detection) | 90% | 81% |

# | ROC-AUC | 0.9996 | — |

# | False Negatives (Missed Defaults) | 4 | 11 |

# | False Positives (Wrong Rejections) | 4 | 15 |





# 

# > ⚠️ \*\*Note:\*\* The 96% V1 accuracy was an artefact of label leakage the model was reverse-engineering a deterministic formula, not learning real patterns. The \*\*87% V2 accuracy is the credible, honest baseline\*\*. See Section 4.1 for full explanation.

# 

# ---

# 

# \## 🧠 4. Deep Technical Reasoning — The Debate

# 

# ---

# 

# \### 4.1 The "Perfect Score" Problem — Why 96% Was a Red Flag

# 

# V1's near-perfect accuracy (96%, ROC-AUC 0.9996) looked impressive but was misleading. The target label was generated as a \*\*deterministic formula\*\* of only 3 features:

# 

# ```python

# score = (Utility\_Bill\_Late\_Days \* 0.4) - (Mobile\_Reload\_Consistency \* 15) + (Existing\_Loans \* 5)

# Loan\_Default = (score > 5)

# ```

# 

# The model was not learning. It was memorising the formula. V2 corrects this by introducing Gaussian noise and multi-factor inter-feature correlations, making the prediction problem genuinely hard. The accuracy drop to 87% is \*\*intentional and desirable\*\*.

# 

# > In credit scoring, a near-perfect model is almost always a sign of \*\*data leakage\*\*; not genuine intelligence.

# 

# ---

# 

# \### 4.2 Threshold Strategy — Beyond the Default 0.5

# 

# The standard XGBoost classification threshold is `0.5`. Lowering it makes the model stricter about flagging risk:

# 

# | Threshold | False Negatives | False Positives | Business Impact |

# |---|---|---|---|

# | 0.3 | 4 | 7 | Too aggressive — over-rejects good applicants |

# | \*\*0.4\*\* | \*\*4\*\* | \*\*5\*\* | ✅ Recommended — balanced risk \& revenue |

# | 0.5 | 4 | 4 | May miss borderline risk in noisy real-world data |

# 

# \*\*Recommended: 0.4 Balanced Threshold\*\*

# 

# While a threshold of 0.3 minimises lender risk on paper, it causes too many creditworthy applicants to be rejected. Since a micro-finance business \*\*generates revenue from loan interest disbursement\*\*, over-rejection directly harms sustainability. The 0.4 threshold controls default risk without starving the business of loan volume.

# 

# ---

# 

# \### 4.3 Repayment Intent vs. Repayment Capacity

# 

# `Community\_Group\_Member` was designed under the assumption that members of Samurdhi, trade associations, or death-donation societies face \*\*social pressure to repay\*\*, making them lower risk. The V2 data supports this directionally — community members default at \*\*19.1%\*\* vs \*\*31.5%\*\* for non-members.

# 

# However, two distinct concepts must not be conflated:

# 

# \- ✅ \*\*Repayment Intent\*\* — wanting to repay to protect social standing

# \- ⚠️ \*\*Repayment Capacity\*\* — actually having the income to repay

# 

# Samurdhi beneficiaries often belong to \*\*lower-income brackets\*\* — precisely where capacity is most constrained. Social pressure and financial ability are separate variables and the model must treat them independently.

# 

# \*\*Recommendation:\*\* Distinguish between \*formal registered groups\* (higher accountability) and \*informal community ties\*, and always pair social capital signals with income-capacity indicators such as `loan\_to\_income\_ratio`.

# 

# ---

# 

# \### 4.4 Dataset Scale — Prototype vs. Production

# 

# | Stage | Records Needed |

# |---|---|

# | Prototype / Demo | 1,000 (current) |

# | Pilot Deployment | 5,000–10,000 |

# | Production | 50,000+ with real repayment history |

# 

# At 1,000 records the model risks \*\*overfitting\*\* — memorising sample-specific patterns rather than generalising to the real applicant population. Cross-validation would likely reveal a lower true accuracy than the reported test scores.

# 

# ---

# 

# \## 💻 5. Tech Stack \& Project Structure

# 

# | Layer | Technology |

# |---|---|

# | ML Engine | XGBoost |

# | Data Processing | Pandas, NumPy, Scikit-learn |

# | Web Interface | Streamlit |

# | Model Serialisation | Joblib |

# 

# ```

# lanka-microfinance-ai/

# │

# ├── app.py                              # Streamlit prediction dashboard

# ├── model\_training.ipynb                # ML pipeline — train / evaluate / save

# ├── data\_generator.ipynb                # Realistic synthetic data generation

# ├── eda\_analysis.ipynb                  # Exploratory data analysis

# │

# ├── data/

# │   ├── lanka\_microfinance\_data.csv     # V1 — synthetic baseline dataset

# │   ├── lanka\_microfinance\_data\_v2.csv  # V2 — realistic dataset with noise

# │   └── microfinance\_model.pkl          # Serialised XGBoost model

# │

# └── requirements.txt

# ```

# 

# ---

# 

# \## ⚙️ 6. Setup \& Installation

# 

# \*\*Clone the repository:\*\*

# ```bash

# git clone https://github.com/your-username/Lanka-Microfinance-AI.git

# cd Lanka-Microfinance-AI

# ```

# 

# \*\*Install dependencies:\*\*

# ```bash

# pip install -r requirements.txt

# ```

# 

# \*\*Run the prediction dashboard:\*\*

# ```bash

# streamlit run app.py

# ```

# 

# > Ensure `microfinance\_model.pkl` is placed inside the `data/` folder before running.

# 

# ---

# 

# \## 🚀 7. Roadmap \& Future Improvements

# 

# \- \*\*Explainable AI (XAI):\*\* Integrate SHAP values to explain \*why\* a loan was rejected — not just \*that\* it was

# \- \*\*Threshold Calibration:\*\* Use precision-recall curves and Optuna Bayesian optimisation to fine-tune threshold per lender risk appetite

# \- \*\*Real-time API Integration:\*\* Connect to telecom APIs for live mobile reload consistency data

# \- \*\*Fairness Audit:\*\* Test for demographic bias across business types, age groups, and income brackets

# \- \*\*Feature Engineering:\*\* Build interaction features — `income\_per\_dependent`, `loan\_to\_income\_ratio`, `late\_days\_per\_loan`

# \- \*\*Mobile Field App:\*\* Lightweight React Native version for field officers collecting applicant data offline

# 

# ---

# 

# \## 👨‍💻 Author

# 

# \*\*Charuka Bandara\*\*  

# IT Undergraduate · ML Enthusiast · Founder of ArtXpert Design

# 

# \[LinkedIn](#www.linkedin.com/in/charuka-mayura)

# 

# ---

# 

# > \*\*Disclaimer:\*\* This is a research prototype. Financial institutions must conduct rigorous stress testing, bias audits, and regulatory review before any real-world deployment.

