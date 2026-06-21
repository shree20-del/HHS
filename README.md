# hhs-care-load-forecasting.

🏥 Predictive Forecasting of Children in HHS Care using Random Forest, Gradient Boosting & ARIMA | 98.14% Accuracy | Streamlit Dashboard | U.S. HHS UAC Program

# Topics / Tags
machine-learning
time-series-forecasting
random-forest
gradient-boosting
arima
streamlit
python
data-science
healthcare-analytics
predictive-analytics
hhs
uac-program
exponential-smoothing
pandas
scikit-learn

# 🏥 Predictive Forecasting of Care Load & Placement Demand
### U.S. Department of Health and Human Services — UAC Program

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Accuracy](https://img.shields.io/badge/Accuracy-98.14%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview
A complete machine learning system that forecasts the number of children
in HHS care up to **30 days in advance** with **98.14% accuracy**.

Built for the **Unaccompanied Alien Children (UAC) Program** to enable
proactive healthcare and child-welfare planning instead of reactive responses.

---

## 🚀 Live Dashboard
👉 **[Open Streamlit App]
([https://shreeee-care-load.streamlit.app/)**

---

## 📊 Models Used & Results

| Model | MAE | RMSE | MAPE | Type |
|---|---|---|---|---|
| ✅ Random Forest | 41.83 | 62.78 | 1.97% | ML |
| ✅ Gradient Boosting | 47.85 | 70.84 | 2.27% | ML |
| ARIMA(5,1,0) | 253.66 | 309.16 | 11.95% | Statistical |
| Exp Smoothing | 286.79 | 389.45 | 12.37% | Statistical |
| Moving Average | 21.31 | 26.43 | 0.95% | Baseline |
| Naive (Lag-1) | 6.62 | 8.24 | 0.29% | Baseline |

---

## 🎯 Key Performance Indicators

| KPI | Value |
|---|---|
| Forecast Accuracy | 98.14% |
| Short-term MAPE (1-7 days) | 1.60% |
| Medium-term MAPE (8-30 days) | 2.37% |
| Forecast Stability Index | 0.88 |
| Dataset Size | 1,068 daily records |
| Forecast Horizons | 7 / 14 / 30 days |

---

## 📁 Project Files

| File | Description |
|---|---|
| `forecasting.ipynb` | 18-step ML pipeline notebook |
| `app.py` | Full Streamlit dashboard (5 tabs) |
| `hhs_data.xlsx` | Dataset — 1,068 daily records |
| `requirements.txt` | Python dependencies |
| `research_paper.docx` | Full technical research report |
| `executive_summary.docx` | Government stakeholder briefing |

---

## 🖥️ Dashboard Features
- 📊 **Overview Tab** — KPI cards, historical chart, net flow, pressure analysis
- 🔍 **Model Comparison Tab** — All 6 models, 90% confidence interval, evaluation table
- 🔮 **Future Forecast Tab** — 7/14/30-day selector, scenario comparison
- 💊 **Discharge Demand Tab** — Placement demand forecast and balance chart
- 📈 **EDA & Insights Tab** — Correlation heatmap, seasonality, decomposition

---

## ⚙️ How to Run Locally

# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/hhs-care-load-forecasting.git
cd hhs-care-load-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py

---

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **ML Models:** Scikit-learn, Statsmodels
- **Dashboard:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📄 License
This project is licensed under the **MIT License** — see LICENSE file for details.

---

##  Acknowledgements
- **Mentor Organization:** Unified Mentor
- **Data Source:** U.S. Department of Health and Human Services
- **Program:** UAC (Unaccompanied Alien Children) Program
  
