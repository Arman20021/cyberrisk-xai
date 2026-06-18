CyberRiskXAI: Machine Learning and Explainable AI-Based Cybersecurity Risk Prediction

Project Overview
CyberRiskXAI is a Streamlit-based web application developed for the thesis topic:
"Machine Learning and Explainable AI-Based Cybersecurity Risk Prediction Among University Students in Bangladesh".

The system allows users to upload cybersecurity awareness survey data in CSV format. It automatically cleans the data, encodes survey answers into numeric values, creates cybersecurity risk scores, trains a Logistic Regression model, evaluates the model, and explains predictions using SHAP and LIME. The website also provides security improvement recommendations based on weak cybersecurity behavior areas.

Main Features
1. CSV Upload
   - Users can upload survey data in CSV format.
   - The app supports both raw survey CSV files and already encoded/ML-ready CSV files.

2. Data Cleaning
   - Removes unnecessary columns.
   - Cleans column names and text values.
   - Handles missing or invalid responses where possible.

3. Answer Encoding
   - Converts cybersecurity survey answers into numeric values.
   - The scoring logic follows:
     1 = risky or weak cybersecurity behavior
     2 = moderate cybersecurity behavior
     3 = safer cybersecurity behavior

4. Risk Score Creation
   - Calculates Cyber Safety Score.
   - Calculates Cyber Risk Score.
   - Labels each respondent as Low Risk, Medium Risk, or High Risk.

5. Machine Learning Model Training
   - Trains a Logistic Regression model.
   - Uses train-test split for evaluation.
   - Applies StandardScaler before training.

6. Model Evaluation
   - Shows Accuracy, Precision, Recall, and F1-score.
   - Shows Classification Report.
   - Shows Confusion Matrix.

7. Explainable AI
   - SHAP is used for global feature importance.
   - LIME is used for local individual prediction explanation.

8. Security Improvement Recommendations
   - Shows the weakest cybersecurity behavior areas.
   - Provides institutional improvement actions.
   - Provides individual student improvement suggestions.
   - Allows downloading security recommendation results.

9. Download Options
   - Processed risk dataset
   - Final ML dataset
   - Classification report
   - Confusion matrix
   - Security recommendations

Technology Stack
- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- SHAP
- LIME
- Matplotlib

Project Files
The project folder should contain:

cyberrisk_xai_app/
    app.py
    requirements.txt
    README.txt

Installation Instructions
1. Open the project folder in VS Code or terminal.

2. Install the required packages:
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py

4. Open the local URL shown in the terminal.

CSV Input Format
The system supports two types of CSV files:

1. Raw Survey CSV
   This file contains original survey questions and text-based answers.

2. ML-Ready CSV
   This file already contains numeric encoded columns such as:
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

Workflow
1. Upload CSV file.
2. Preview raw data.
3. Clean and encode survey answers.
4. Create cybersecurity risk scores.
5. Generate Low, Medium, and High Risk labels.
6. Train Logistic Regression model.
7. Evaluate model performance.
8. Generate SHAP global explanation.
9. Generate LIME local explanation.
10. Generate security improvement recommendations.
11. Download processed outputs.

Risk Level Meaning
Low Risk:
The respondent demonstrates safer cybersecurity practices.

Medium Risk:
The respondent has moderate cybersecurity awareness but still needs improvement.

High Risk:
The respondent shows risky cybersecurity behavior and needs immediate awareness support.

Explainable AI Interpretation
SHAP:
SHAP identifies the most influential features affecting the model prediction. It helps explain which cybersecurity behaviors contribute most to risk prediction.

LIME:
LIME explains one individual prediction. It shows why a selected respondent was predicted as Low Risk, Medium Risk, or High Risk.

Security Recommendation Section
The recommendation section identifies weak cybersecurity behavior areas and provides practical improvement suggestions, such as:
- Use unique passwords.
- Enable Two-Factor Authentication.
- Avoid suspicious links.
- Avoid cracked software.
- Use password managers.
- Verify HTTPS before entering sensitive information.
- Avoid sensitive work on public Wi-Fi.
- Improve phishing awareness.

Privacy and Ethical Note
This system should be used with anonymized survey data. Personal identifiers such as student names, phone numbers, emails, IDs, or addresses should not be uploaded unless proper permission and ethical approval are available.

Limitations
- The model performance depends on the quality of the uploaded survey data.
- If the dataset is highly imbalanced, minority risk classes may be harder to predict.
- SHAP and LIME explanations may take time to generate for large datasets.
- The system is designed for research and educational use, not as a replacement for professional cybersecurity audits.

Future Improvements
- Add user login and admin dashboard.
- Add PDF report generation.
- Add database support.
- Add department-wise and year-wise risk trend analysis.
- Add comparison between multiple ML models.
- Add automatic email report generation.
- Add more advanced XAI visualizations.

Thesis Use
This project demonstrates the practical implementation of a machine learning and explainable AI-based cybersecurity risk prediction system. It supports data preprocessing, risk scoring, model training, model evaluation, explainability, and actionable security recommendations through a single web-based dashboard.

Author
Prepared for thesis implementation:
Machine Learning and Explainable AI-Based Cybersecurity Risk Prediction Among University Students in Bangladesh.
