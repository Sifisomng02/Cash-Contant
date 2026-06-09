# 💡 CashContant

> **Tracking South Africa's cash dependency — and the risks that come with it.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-1.1.5-0077B5?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Problem Statement

Despite rapid growth in digital payments, **physical banknote usage in South Africa remains structurally high**:

- **82%** of South Africans still rely on cash for everyday transactions *(SARB, 2023)*
- **58%** of all payments under R100 are made in cash *(SARB Payments Report, 2023)*
- **R1.2 Trillion** was withdrawn from ATMs in 2022 alone
- Cash-in-transit heists, ATM bombings, and card skimming cost South Africans billions annually

CashContant analyses *why* cash demand persists and *what risks* that demand creates — combining macro SARB data, World Bank ATM statistics, and personal withdrawal behaviour in a single interactive dashboard.

---

## 🔍 Key Findings

- **M0 (cash in circulation) has grown consistently** year-on-year, even as digital payment volumes rise — indicating structural, not declining, cash dependency
- **ATM density peaked and stabilised** rather than declining, confirming the cash infrastructure remains entrenched
- **End-of-month salary cycles** are clearly visible in withdrawal patterns — most South Africans withdraw large amounts on the 25th–31st
- **After-hours ATM usage (00h–04h)** represents a quantifiable fraud and personal safety risk window
- **Velocity and travel-based anomaly detection** can flag potentially fraudulent card usage with no ML training required

---

## 🖥️ App Features

| Page | Description |
|---|---|
| 📈 My Dashboard | Personal KPIs, monthly habits, national M0 trend, ATM density |
| 🗺️ Branch Hotspots | Interactive map of your most-visited branches |
| 🔮 Withdrawal Forecast | Prophet + Linear Regression forecast with RMSE/MAE comparison |
| 🛡️ Security & Risk | Fraud detection scan with per-transaction anomaly scoring (0–100) |
| 🗂️ Raw Data Explorer | Inspect all loaded datasets directly |

---

## 🧠 Technical Highlights

**`models/model_demand.py`**
- Linear Regression baseline forecast with RMSE and MAE evaluation
- Daily resampling pipeline compatible with Prophet
- Model comparison framework: LR vs Prophet side-by-side

**`models/model_security.py`**
- 5-layer fraud detection: rapid withdrawals, impossible travel, after-hours, high-value outliers, negative amounts
- Weighted composite **anomaly score (0–100)** per transaction
- Risk classification: Normal / Low / Medium / High

**`notebooks/EDA.ipynb`**
- Full exploratory analysis on SARB M0, World Bank ATM density, and personal ATM data
- Documented findings with annotated visualisations

---

## 🗂️ Project Structure

```
CashContant/
├── app.py                  # Main Streamlit application
├── config.py               # Centralised paths, constants, thresholds
├── requirements.txt        # Pinned dependencies
├── .gitignore
├── README.md
├── models/
│   ├── model_demand.py     # Forecasting (LR + Prophet integration)
│   └── model_security.py  # Fraud detection + anomaly scoring
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis
├── data/
│   └── README.md           # Data sources & column schemas
└── images/
    └── logo.png
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/CashContant.git
cd CashContant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your data
Place the required files in the `data/` folder — see [`data/README.md`](data/README.md) for the full list of sources and expected column schemas.

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 Data Sources

| Dataset | Source | Link |
|---|---|---|
| Cash in Circulation (M0) | South African Reserve Bank | [SARB Quarterly Bulletin](https://www.resbank.co.za) |
| ATM Density | World Bank Open Data | [FB.ATM.TOTL.P5](https://data.worldbank.org/indicator/FB.ATM.TOTL.P5) |
| Money & Banking Report | SARB | [SARB Publications](https://www.resbank.co.za/en/home/publications) |
| Household Expenditure | Statistics South Africa | [StatsSA](https://www.statssa.gov.za) |

---

## 👤 Author

**Sifiso Mnguni** — Data Analyst  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/your-username)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
