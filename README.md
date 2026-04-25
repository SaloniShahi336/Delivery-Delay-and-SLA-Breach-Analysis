# E-Commerce Delivery Delay & SLA Breach Prediction

> End-to-end ML pipeline predicting delivery delays and SLA breaches across 125,000+ orders — achieving 96.4% accuracy and 0.93 ROC-AUC with 59+ engineered features.

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=mysql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📌 Business Problem

Late deliveries destroy customer trust and create cascading supply chain costs. This project builds a **predictive system that flags at-risk orders before an SLA breach occurs**, giving operations teams time to intervene.

Key questions:
- Which orders are most likely to miss their delivery SLA?
- What operational and geographic factors drive delays?
- Can we predict breaches with enough lead time to act?

---

## 📊 Data

| Attribute | Detail |
|---|---|
| Total orders | 125,000+ |
| Source datasets | 5 interconnected datasets (orders, shipments, carriers, geography, SLA definitions) |
| Key fields | Order timestamp, carrier, origin/destination, SLA window, actual delivery date |

---

## ⚙️ Methodology

1. **Data Pipeline** — Joined and validated 5 source datasets; resolved inconsistencies in carrier codes, timestamp formats, and SLA definitions across regions
2. **Feature Engineering** — Constructed 59+ features capturing temporal patterns (order day, hour, week-of-year), geographic risk, carrier performance history, product category, and distance-based metrics
3. **Exploratory Analysis** — Identified delay hotspots by region, carrier, and time window; analyzed SLA breach rates by product type
4. **Model Development** — Trained and compared Logistic Regression, Random Forest, and XGBoost; tuned hyperparameters via cross-validation
5. **Evaluation** — Assessed models on accuracy, ROC-AUC, precision/recall trade-offs; selected XGBoost as final model
6. **Deployment** — Packaged prediction pipeline for batch scoring of new orders

---

## 📈 Key Results

| Metric | Result |
|---|---|
| Accuracy | **96.4%** |
| ROC-AUC | **0.93** |
| Features engineered | **59+** |
| Orders analyzed | **125,000+** |

XGBoost outperformed baseline models by 8+ percentage points on AUC, with feature importance analysis confirming that carrier performance history and destination geography were the strongest delay predictors.

---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/saloni-shahi/delivery-delay-prediction.git
cd delivery-delay-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/pipeline.py

# Or step through the notebook
jupyter notebook notebooks/delivery_delay_modeling.ipynb
```

**Folder structure:**
```
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and modeling notebooks
├── src/                # Pipeline scripts
├── models/             # Saved model artifacts
└── requirements.txt
```

---

## 💡 Learnings

- Feature engineering contributed more to model performance than algorithm selection — time invested in domain-informed features (carrier history, geographic risk scores) yielded the biggest accuracy gains
- XGBoost's feature importance outputs were essential for translating model results into operational recommendations: "flag orders shipped via Carrier X to Region Y on Mondays"
- Building the pipeline end-to-end (ingestion → features → model → scoring) is a different skill than just fitting a model — understanding data contracts between stages matters

---

