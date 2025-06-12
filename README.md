# credit-card-fraud-detection
A machine learning project that classifies credit card transactions as fraudulent or legitimate.

## 📌 Project Overview

- 🔍 **Problem Statement**: Detect whether a credit card transaction is fraudulent or legitimate based on transaction and customer profile data.
- 🧠 **Model Used**: Random Forest Classifier
- 📊 **Accuracy Achieved**: 99.77%
- ⚠️ **Challenge**: Class imbalance (fraudulent cases are extremely rare)

---

## 🗂️ Dataset

The dataset used contains over 500,000 transactions with the following fields:

- `trans_date_trans_time` – Transaction timestamp  
- `merchant`, `category`, `amt`, `city`, `state`, `zip`, `lat`, `long` – Transaction details  
- `gender`, `dob`, `job`, `merch_lat`, `merch_long` – Customer info  
- `is_fraud` – Target variable (1 = fraud, 0 = legitimate)

> 📁 File used: `fraudTest.csv`

---

## 📈 Model Performance

| Metric      | Value    |
|-------------|----------|
| Accuracy    | 99.77%   |
| Precision   | 88%      |
| Recall      | 48%      |
| F1-Score    | 62%      |

### 🧾 Confusion Matrix

---

## ⚙️ How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/sawrypathak/credit-card-fraud-detection.git
cd credit-card-fraud-detection

