# 🛡️ CyberRiskXAI

## Machine Learning and Explainable AI-Based Cybersecurity Risk Prediction

CyberRiskXAI is a **Streamlit-based web application** developed for the thesis topic:

> **Machine Learning and Explainable AI-Based Cybersecurity Risk Prediction Among University Students in Bangladesh**

The system allows users to upload cybersecurity awareness survey data in CSV format. It automatically cleans the data, encodes survey answers into numeric values, creates cybersecurity risk scores, trains a **Logistic Regression** model, evaluates the model, and explains predictions using **SHAP** and **LIME**. It also provides practical cybersecurity improvement recommendations based on weak security behavior areas.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [CSV Input Format](#-csv-input-format)
- [System Workflow](#-system-workflow)
- [Risk Level Meaning](#-risk-level-meaning)
- [Explainable AI](#-explainable-ai)
- [Security Improvement Recommendations](#-security-improvement-recommendations)
- [Downloadable Outputs](#-downloadable-outputs)
- [Privacy and Ethics](#-privacy-and-ethics)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Thesis Relevance](#-thesis-relevance)

---

## 📖 Project Overview

CyberRiskXAI is designed to transform raw cybersecurity awareness survey data into meaningful cybersecurity risk insights.

The system performs:

1. CSV upload
2. Data cleaning
3. Answer encoding
4. Risk score creation
5. Risk level labeling
6. Logistic Regression model training
7. Model evaluation
8. SHAP-based global explanation
9. LIME-based local explanation
10. Security improvement recommendation generation

This makes the system useful for universities, researchers, and administrators who want to analyze cybersecurity risk behavior among students.

---

## 🚀 Key Features

### ✅ CSV Upload

- Upload cybersecurity awareness survey data in CSV format.
- Supports both:
  - Raw survey CSV
  - Already encoded / ML-ready CSV

### ✅ Data Cleaning

- Cleans column names.
- Removes unnecessary columns.
- Handles missing or invalid responses where possible.

### ✅ Answer Encoding

Survey responses are converted into numerical values:

| Score | Meaning |
|---:|---|
| `1` | Risky or weak cybersecurity behavior |
| `2` | Moderate cybersecurity behavior |
| `3` | Safer cybersecurity behavior |

### ✅ Risk Score Creation

The system calculates:

- **Cyber Safety Score**
- **Cyber Risk Score**
- **Risk Level**
- **Risk Level Numeric**

### ✅ Machine Learning Model

The system trains a **Logistic Regression** model using the processed dataset.

### ✅ Model Evaluation

The dashboard displays:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### ✅ Explainable AI

The system uses:

- **SHAP** for global feature importance
- **LIME** for local individual prediction explanation

### ✅ Security Recommendations

The system identifies weak cybersecurity areas and suggests how students or institutions can improve security behavior.

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application framework |
| Pandas | Data processing |
| NumPy | Numerical computation |
| scikit-learn | Machine learning model training and evaluation |
| Plotly | Interactive charts |
| SHAP | Global explainable AI |
| LIME | Local explainable AI |
| Matplotlib | Supporting visualization |

---

## 📁 Project Structure

```bash
cyberrisk_xai_app/
│
├── app.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/cyberrisk-xai.git
```

Then enter the project folder:

```bash
cd cyberrisk-xai
```

Or, if you are using a local folder:

```bash
cd F:\cyberrisk_xai_app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Streamlit app:

```bash
streamlit run app.py
```

After running the command, Streamlit will show a local URL similar to:

```bash
http://localhost:8501
```

Open the link in your browser.

---

## 📄 CSV Input Format

The system supports two types of CSV files.

---

### 1. Raw Survey CSV

This file contains original survey questions and text-based answers collected from students.

Example raw columns may include:

- Gender
- Age
- Department
- Year of Study
- Password reuse behavior
- 2FA usage
- Unknown link clicking behavior
- Phishing awareness
- Password sharing behavior
- Public Wi-Fi usage
- Antivirus usage
- Password manager usage
- Software authenticity behavior
- Separate device usage

---

### 2. ML-Ready CSV

The system also supports an already encoded dataset with columns such as:

```text
Q5_password_reuse
Q6_2FA
Q7_unknown_links
Q8_phishing_awareness
Q9_password_share
Q10_public_wifi
Q11_antivirus
Q12_unique_passwords
Q13_privacy_policy
Q14_https_check
Q15_online_threat_learning
Q16_password_manager
Q17_change_passwords
Q18_software_authenticity
Q19_separate_devices
Gender_Female
Gender_Male
Age_18-23
Age_Below 18
Age_above 23
Department_BBA
Department_CSE
Department_EEE
Department_Others
Year_1st Year
Year_2nd Year
Year_3rd Year
Year_Final Year
Risk_Level_Numeric
```

---

## 🔁 System Workflow

```text
Upload CSV
    ↓
Data Cleaning
    ↓
Answer Encoding
    ↓
Risk Score Creation
    ↓
Risk Level Labeling
    ↓
Train Logistic Regression Model
    ↓
Evaluate Model Performance
    ↓
Generate SHAP Explanation
    ↓
Generate LIME Explanation
    ↓
Generate Security Recommendations
    ↓
Download Results
```

---

## 🚦 Risk Level Meaning

| Risk Level | Numeric Label | Meaning |
|---|---:|---|
| Low Risk | `0` | Student demonstrates safer cybersecurity behavior |
| Medium Risk | `1` | Student has moderate awareness but still needs improvement |
| High Risk | `2` | Student shows risky cybersecurity behavior and needs immediate support |

---

## 🤖 Machine Learning Model

The system uses **Logistic Regression** as the main prediction model.

### Why Logistic Regression?

- It performed best during model comparison.
- It works well with structured numerical survey data.
- It is interpretable and suitable for academic explanation.
- It can be combined with SHAP and LIME for explainable AI.

---

## 📊 Model Evaluation Metrics

The system evaluates the model using:

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correctness of predicted risk classes |
| Recall | Ability to find actual risk classes |
| F1-score | Balance between precision and recall |

The dashboard also shows:

- Classification Report
- Confusion Matrix

---

## 🧠 Explainable AI

Explainable AI is included to make the model transparent and understandable.

---

### SHAP Explanation

SHAP is used for **global explanation**.

It answers:

> Which features influence the model prediction the most?

Example important features may include:

- Password reuse
- Password manager usage
- Privacy policy reading behavior
- Software authenticity
- Public Wi-Fi usage
- 2FA usage

---

### LIME Explanation

LIME is used for **local explanation**.

It answers:

> Why was this specific student predicted as Low Risk, Medium Risk, or High Risk?

LIME explains one selected student’s prediction by showing the most influential features for that individual case.

---

## 🛠️ Security Improvement Recommendations

The system generates cybersecurity improvement suggestions based on weak behavior areas.

Example recommendations include:

- Use unique passwords for important accounts.
- Enable Two-Factor Authentication.
- Avoid clicking suspicious links.
- Avoid using cracked software.
- Use a password manager.
- Verify HTTPS before entering sensitive information.
- Avoid sensitive activities on public Wi-Fi.
- Improve phishing and scam awareness.
- Stop sharing passwords with others.

---

## ⬇️ Downloadable Outputs

The system allows users to download:

| Output File | Description |
|---|---|
| `processed_risk_dataset.csv` | Dataset with risk score and risk level |
| `final_ml_dataset.csv` | Final numeric dataset used for ML |
| `classification_report.csv` | Model classification performance |
| `confusion_matrix.csv` | Confusion matrix result |
| `security_recommendations.csv` | Security improvement recommendation report |

---

## 🔐 Privacy and Ethics

This system should be used with anonymized survey data.

Users should avoid uploading personal identifiers such as:

- Student names
- Phone numbers
- Email addresses
- Student IDs
- Home addresses
- Any sensitive personal information

The system is intended for academic research and awareness improvement purposes.

---

## ⚠️ Limitations

- Model performance depends on dataset quality.
- Highly imbalanced datasets may affect minority class prediction.
- SHAP and LIME may take time for large datasets.
- The system is not a replacement for a professional cybersecurity audit.
- Risk labels are based on survey scoring logic, so the model learns patterns from the defined scoring system.

---

## 🔮 Future Improvements

Possible future improvements include:

- Add user login system.
- Add admin dashboard.
- Add database support.
- Add PDF report generation.
- Add department-wise risk trend analysis.
- Add year-wise cybersecurity behavior comparison.
- Add comparison between multiple ML models.
- Add automatic email report generation.
- Add more advanced SHAP visualizations.
- Deploy the system publicly using Streamlit Community Cloud.

---

## 🎓 Thesis Relevance

This project demonstrates the practical implementation of:

- Survey-based cybersecurity risk analysis
- Data preprocessing
- Risk score creation
- Machine learning model training
- Model evaluation
- Explainable AI
- Security recommendation generation
- Web-based dashboard deployment

It supports the thesis objective of using **Machine Learning** and **Explainable AI** to predict and explain cybersecurity risk among university students in Bangladesh.

---

## 👤 Author

Prepared for thesis implementation:

**Machine Learning and Explainable AI-Based Cybersecurity Risk Prediction Among University Students in Bangladesh**

---

## 📌 Project Status

```text
Status: Thesis Prototype / Academic Research Project
```
