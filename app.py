

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import lime
import lime.lime_tabular
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CyberRiskXAI",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CUSTOM DESIGN
# ============================================================
st.markdown(
    """
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #0f766e 100%);
        padding: 38px;
        border-radius: 24px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
    }

    .hero-title {
        font-size: 44px;
        font-weight: 850;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 18px;
        line-height: 1.7;
        color: #e0f2fe;
        max-width: 980px;
    }

    .hero-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        color: #ffffff;
        padding: 8px 14px;
        border-radius: 999px;
        margin-right: 8px;
        margin-top: 18px;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 5px;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        min-height: 118px;
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 31px;
        color: #0f172a;
        font-weight: 850;
    }

    .info-box {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 16px 18px;
        border-radius: 12px;
        color: #1e3a8a;
        margin-bottom: 16px;
        font-weight: 600;
    }

    .success-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-weight: 800;
        font-size: 14px;
    }

    .warning-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        background: #fef9c3;
        color: #854d0e;
        font-weight: 800;
        font-size: 14px;
    }

    .danger-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 800;
        font-size: 14px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
        line-height: 1.5;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 12px 16px;
        border: 1px solid #e2e8f0;
        font-weight: 800;
    }

    .stTabs [aria-selected="true"] {
        background: #1e3a8a !important;
        color: white !important;
        border-color: #1e3a8a !important;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #2563eb;
        padding: 22px;
        border-radius: 18px;
    }

    .download-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 14px;
    }
    .rec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #0f766e;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    .rec-title {
        font-size: 18px;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .rec-issue {
        color: #991b1b;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .rec-action {
        color: #166534;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .rec-text {
        color: #334155;
        line-height: 1.55;
        font-size: 15px;
    }

    .priority-high {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .priority-medium {
        display: inline-block;
        background: #fef9c3;
        color: #854d0e;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 8px;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UI HELPERS
# ============================================================
def section_header(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def custom_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk):
    if risk == "Low Risk":
        return '<span class="success-badge">Low Risk</span>'
    if risk == "Medium Risk":
        return '<span class="warning-badge">Medium Risk</span>'
    return '<span class="danger-badge">High Risk</span>'


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# HERO + SIDEBAR
# ============================================================
st.markdown(
    """
 
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ CyberRiskXAI")
    st.caption("Thesis Implementation Dashboard")

    st.markdown("---")
    st.markdown("### 🔁 Workflow")
    st.write("1. Upload CSV")
    st.write("2. Clean Data")
    st.write("3. Encode Answers")
    st.write("4. Create Risk Scores")
    st.write("5. Train Logistic Regression")
    st.write("6. Evaluate Model")
    st.write("7. Explain with SHAP/LIME")

    st.markdown("---")
    st.markdown("### ✅ Current Model")
    st.success("Logistic Regression")

    st.markdown("### 🧠 XAI Methods")
    st.info("SHAP + LIME")

    st.markdown("---")
    st.caption("Tip: Use the same survey question format as your original thesis dataset.")


# ============================================================
# SURVEY CONFIG
# ============================================================
score_map = {
    "Q5_password_reuse": {
        "Maximum are reused": 1,
        "Mostly unique": 2,
        "Absolute unique": 3,
    },
    "Q6_2FA": {
        "Always if avilable": 3,
        "Feels like a hassle": 2,
        "Don't Know About 2FA": 1,
    },
    "Q7_unknown_links": {
        "Often": 1,
        "Sometimes": 2,
        "Never": 3,
    },
    "Q9_password_share": {
        "Yes, I think they have rights to know": 1,
        "To people with close relationship": 2,
        "No, no matter how close they are": 3,
    },
    "Q10_public_wifi": {
        "Don't know about public wifi": 1,
        "Sometimes if essential": 2,
        "Never": 3,
    },
    "Q11_antivirus": {
        "No": 1,
        "Not all devices": 2,
        "Yes": 3,
    },
    "Q12_unique_passwords": {
        "All passwords are similar": 1,
        "Some are unique some are similar": 2,
        "unique for different account": 3,
    },
    "Q13_privacy_policy": {
        "Avoid it": 1,
        "Sometimes": 2,
        "Always": 3,
    },
    "Q14_https_check": {
        "Don't seems important": 1,
        "Sometimes": 2,
        "Always": 3,
    },
    "Q15_online_threat_learning": {
        "if i got stuck with anything": 1,
        "whenever such contents apper": 2,
        "Try to always": 3,
    },
    "Q16_password_manager": {
        "No": 1,
        "Sometimes": 2,
        "Always": 3,
    },
    "Q17_change_passwords": {
        "Not for all platform": 1,
        "Sometimes": 2,
        "Always": 3,
    },
    "Q18_software_authenticity": {
        "Uses crack mostly": 1,
        "Gets from friends and family": 2,
        "Buys from Original company": 3,
    },
    "Q19_separate_devices": {
        "All in one device": 1,
        "Try to use separate device": 3,
    },
}

standard_columns = [
    "Gender",
    "Age",
    "Department",
    "Year",
    "Q5_password_reuse",
    "Q6_2FA",
    "Q7_unknown_links",
    "Q8_phishing_awareness",
    "Q9_password_share",
    "Q10_public_wifi",
    "Q11_antivirus",
    "Q12_unique_passwords",
    "Q13_privacy_policy",
    "Q14_https_check",
    "Q15_online_threat_learning",
    "Q16_password_manager",
    "Q17_change_passwords",
    "Q18_software_authenticity",
    "Q19_separate_devices",
    "Self_Awareness_Level",
]

class_names = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}


# ============================================================
# SECURITY RECOMMENDATION BANK
# ============================================================
recommendation_bank = {
    "Q5_password_reuse": {
        "title": "Reduce Password Reuse",
        "issue": "Students are reusing passwords across multiple accounts.",
        "action": "Use a unique password for every important account.",
        "details": "Encourage students to create different passwords for email, university portals, banking, and social media. Reused passwords increase risk because one leaked account can expose many other accounts.",
    },
    "Q6_2FA": {
        "title": "Enable Two-Factor Authentication",
        "issue": "Some students do not regularly use 2FA.",
        "action": "Enable 2FA on email, university, banking, and social accounts.",
        "details": "Two-Factor Authentication adds an extra security layer even if a password is stolen. Prioritize email accounts because email is often used for password recovery.",
    },
    "Q7_unknown_links": {
        "title": "Avoid Suspicious Links",
        "issue": "Students may click links from unknown emails or messages.",
        "action": "Verify the sender and link before clicking.",
        "details": "Teach students to check sender address, URL spelling, and message urgency. Suspicious links can lead to phishing, credential theft, or malware.",
    },
    "Q8_phishing_awareness": {
        "title": "Improve Phishing Awareness",
        "issue": "Phishing and scam awareness needs improvement.",
        "action": "Arrange phishing awareness training and short quizzes.",
        "details": "Use simple examples of fake login pages, urgent messages, prize scams, and academic account phishing attempts.",
    },
    "Q9_password_share": {
        "title": "Stop Password Sharing",
        "issue": "Some students share passwords with friends or family.",
        "action": "Do not share passwords with anyone.",
        "details": "Promote account privacy. If access must be shared, use official delegation or account recovery options instead of sharing passwords.",
    },
    "Q10_public_wifi": {
        "title": "Avoid Sensitive Work on Public Wi-Fi",
        "issue": "Students may use public Wi-Fi for sensitive activities.",
        "action": "Avoid banking, exam, and personal account access on unsafe public Wi-Fi.",
        "details": "Use trusted networks for sensitive work. If public Wi-Fi is unavoidable, avoid entering sensitive information and check website security carefully.",
    },
    "Q11_antivirus": {
        "title": "Use Updated Security Software",
        "issue": "Some devices may not have antivirus or security protection.",
        "action": "Install and regularly update trusted security software.",
        "details": "Security software helps detect malware, suspicious downloads, and harmful files. Students should also keep the operating system updated.",
    },
    "Q12_unique_passwords": {
        "title": "Use Unique Passwords for Important Accounts",
        "issue": "Important accounts may have similar passwords.",
        "action": "Use unique passwords for email, university portal, and financial accounts.",
        "details": "Important accounts should not share the same or similar passwords. This reduces damage if one account is compromised.",
    },
    "Q13_privacy_policy": {
        "title": "Check App Privacy and Permissions",
        "issue": "Students often avoid reading privacy policies or permissions.",
        "action": "Review app permissions before installing.",
        "details": "Students should check whether an app asks for unnecessary access to contacts, storage, microphone, camera, or location.",
    },
    "Q14_https_check": {
        "title": "Verify Website Security",
        "issue": "Students may not check HTTPS before entering sensitive information.",
        "action": "Check HTTPS and domain spelling before login or payment.",
        "details": "Fake websites often use similar-looking domains. Students should verify the lock icon, HTTPS, and correct spelling of the website address.",
    },
    "Q15_online_threat_learning": {
        "title": "Build Continuous Cybersecurity Learning",
        "issue": "Some students only learn about threats after facing a problem.",
        "action": "Encourage regular short awareness content.",
        "details": "Universities can share monthly short posts, videos, or workshops about phishing, password safety, scams, and safe browsing.",
    },
    "Q16_password_manager": {
        "title": "Use a Password Manager",
        "issue": "Students may not use password managers.",
        "action": "Use a trusted password manager to store unique passwords.",
        "details": "Password managers help students create and store strong unique passwords without memorizing all of them.",
    },
    "Q17_change_passwords": {
        "title": "Review and Update Passwords",
        "issue": "Some students do not review or update passwords.",
        "action": "Change passwords after suspected compromise and review them periodically.",
        "details": "Students should immediately change passwords after suspicious activity, password leaks, or account compromise.",
    },
    "Q18_software_authenticity": {
        "title": "Avoid Cracked Software",
        "issue": "Using cracked software creates malware and privacy risk.",
        "action": "Install software only from trusted and official sources.",
        "details": "Cracked software may contain malware or hidden data-stealing tools. Encourage free/open-source or student-licensed alternatives.",
    },
    "Q19_separate_devices": {
        "title": "Separate Sensitive Activities",
        "issue": "All sensitive and casual activities may happen on one device/profile.",
        "action": "Use a separate browser profile or device for banking, exams, and important work.",
        "details": "If a separate device is not possible, use separate browser profiles and avoid installing risky extensions on the profile used for sensitive work.",
    },
}


# ============================================================
# DATA PROCESSING FUNCTIONS
# ============================================================
def clean_text_columns(df):
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        df[col] = df[col].replace(["nan", "NaN", ""], np.nan)

    return df


def prepare_raw_csv(df):
    df = clean_text_columns(df)

    drop_cols = []
    for col in df.columns:
        if col.lower().startswith("column"):
            drop_cols.append(col)

    if "Score" in df.columns:
        drop_cols.append("Score")

    df = df.drop(columns=drop_cols, errors="ignore")

    if len(df.columns) == len(standard_columns):
        df.columns = standard_columns
    elif len(df.columns) == len(standard_columns) - 1:
        df.columns = standard_columns[:-1]
    else:
        st.error("CSV column count does not match the expected survey format.")
        st.write("Expected columns:", len(standard_columns), "or", len(standard_columns) - 1)
        st.write("Found columns:", len(df.columns))
        st.write(df.columns.tolist())
        st.stop()

    return df


def encode_survey(df):
    df_encoded = df.copy()

    for col, mapping in score_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

    if "Q8_phishing_awareness" in df_encoded.columns:
        df_encoded["Q8_phishing_awareness"] = pd.to_numeric(
            df_encoded["Q8_phishing_awareness"],
            errors="coerce",
        )

        df_encoded["Q8_phishing_awareness"] = df_encoded["Q8_phishing_awareness"].map(
            {
                1: 1,
                2: 1,
                3: 2,
                4: 3,
                5: 3,
            }
        )

    return df_encoded


def create_risk_scores(df_encoded):
    df_encoded = df_encoded.copy()

    possible_score_columns = [
        "Q5_password_reuse",
        "Q6_2FA",
        "Q7_unknown_links",
        "Q8_phishing_awareness",
        "Q9_password_share",
        "Q10_public_wifi",
        "Q11_antivirus",
        "Q12_unique_passwords",
        "Q13_privacy_policy",
        "Q14_https_check",
        "Q15_online_threat_learning",
        "Q16_password_manager",
        "Q17_change_passwords",
        "Q18_software_authenticity",
        "Q19_separate_devices",
    ]

    active_score_columns = []

    for col in possible_score_columns:
        if col in df_encoded.columns:
            if df_encoded[col].isnull().all():
                continue

            df_encoded[col] = df_encoded[col].fillna(2)
            active_score_columns.append(col)

    if len(active_score_columns) == 0:
        st.error("No valid scoring columns were found.")
        st.stop()

    df_encoded["Cyber_Safety_Score"] = df_encoded[active_score_columns].sum(axis=1)

    min_score = len(active_score_columns)
    max_score = len(active_score_columns) * 3

    df_encoded["Cyber_Risk_Score"] = (max_score + 1) - df_encoded["Cyber_Safety_Score"]

    high_risk_upper = min_score + ((max_score - min_score) / 3)
    medium_risk_upper = min_score + 2 * ((max_score - min_score) / 3)

    def assign_risk_level(score):
        if score <= high_risk_upper:
            return "High Risk"
        if score <= medium_risk_upper:
            return "Medium Risk"
        return "Low Risk"

    df_encoded["Risk_Level"] = df_encoded["Cyber_Safety_Score"].apply(assign_risk_level)

    risk_map = {
        "Low Risk": 0,
        "Medium Risk": 1,
        "High Risk": 2,
    }

    df_encoded["Risk_Level_Numeric"] = df_encoded["Risk_Level"].map(risk_map)

    scoring_info = {
        "active_score_columns": active_score_columns,
        "min_score": min_score,
        "max_score": max_score,
        "high_risk_upper": high_risk_upper,
        "medium_risk_upper": medium_risk_upper,
    }

    return df_encoded, scoring_info


def create_ml_dataset(scored_df):
    df_ml = scored_df.copy()

    demographic_cols = [col for col in ["Gender", "Age", "Department", "Year"] if col in df_ml.columns]

    df_ml = pd.get_dummies(
        df_ml,
        columns=demographic_cols,
        drop_first=False,
    )

    leakage_cols = [
        "Self_Awareness_Level",
        "Cyber_Safety_Score",
        "Cyber_Risk_Score",
        "Risk_Level",
    ]

    df_ml = df_ml.drop(columns=leakage_cols, errors="ignore")

    bool_cols = df_ml.select_dtypes(include=["bool"]).columns
    df_ml[bool_cols] = df_ml[bool_cols].astype(int)

    df_ml = df_ml.select_dtypes(include=["number"])

    if "Risk_Level_Numeric" not in df_ml.columns:
        st.error("Risk_Level_Numeric was not created correctly.")
        st.stop()

    return df_ml


def train_logistic_regression(ml_dataset):
    X = ml_dataset.drop("Risk_Level_Numeric", axis=1)
    y = ml_dataset["Risk_Level_Numeric"]

    if y.nunique() < 2:
        st.error("Model training requires at least two risk classes.")
        st.stop()

    stratify_value = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    lr_model.fit(X_train_scaled, y_train)

    y_pred = lr_model.predict(X_test_scaled)

    result = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    return lr_model, scaler, X_train, X_test, y_train, y_test, y_pred, result


def predict_proba_original(data, model, scaler, feature_names):
    data_df = pd.DataFrame(data, columns=feature_names)
    data_scaled = scaler.transform(data_df)
    return model.predict_proba(data_scaled)



# ============================================================
# RECOMMENDATION FUNCTIONS
# ============================================================
def build_recommendation_table(scored_df):
    available_cols = [
        col for col in recommendation_bank.keys()
        if col in scored_df.columns and pd.api.types.is_numeric_dtype(scored_df[col])
    ]

    rows = []
    for col in available_cols:
        avg_score = scored_df[col].mean()
        weak_count = int((scored_df[col] <= 2).sum())
        weak_percent = (weak_count / len(scored_df)) * 100 if len(scored_df) else 0

        rec = recommendation_bank[col]
        rows.append({
            "Feature": col,
            "Topic": rec["title"],
            "Average Score": round(avg_score, 3),
            "Weak/Moderate Responses": weak_count,
            "Weak/Moderate %": round(weak_percent, 2),
            "Main Improvement": rec["action"],
        })

    recommendation_df = pd.DataFrame(rows)

    if recommendation_df.empty:
        return recommendation_df

    recommendation_df = recommendation_df.sort_values(
        by=["Average Score", "Weak/Moderate %"],
        ascending=[True, False],
    )

    return recommendation_df


def get_individual_recommendations(student_row, max_items=6):
    items = []

    for col, rec in recommendation_bank.items():
        if col in student_row.index and pd.notna(student_row[col]):
            try:
                value = float(student_row[col])
            except Exception:
                continue

            if value <= 2:
                priority = "High Priority" if value <= 1 else "Medium Priority"
                items.append({
                    "Feature": col,
                    "Score": value,
                    "Priority": priority,
                    "Topic": rec["title"],
                    "Issue": rec["issue"],
                    "Recommended Action": rec["action"],
                    "Details": rec["details"],
                })

    priority_order = {"High Priority": 0, "Medium Priority": 1}
    items = sorted(items, key=lambda x: (priority_order.get(x["Priority"], 2), x["Score"]))

    return items[:max_items]


def recommendation_card(item):
    priority_class = "priority-high" if item.get("Priority", "High Priority") == "High Priority" else "priority-medium"
    priority_text = item.get("Priority", "High Priority")

    return f"""
    <div class="rec-card">
        <span class="{priority_class}">{priority_text}</span>
        <div class="rec-title">{item['Topic']}</div>
        <div class="rec-issue">Problem: {item['Issue']}</div>
        <div class="rec-action">How to improve: {item['Recommended Action']}</div>
        <div class="rec-text">{item['Details']}</div>
    </div>
    """


# ============================================================
# FILE UPLOAD
# ============================================================
section_header(
    "📁 Upload Survey Dataset",
    "Upload your raw survey CSV file. The system will automatically clean, encode, score, train, and explain the model.",
)

uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
            Upload a CSV file to start the cybersecurity risk prediction workflow.
            Make sure the CSV follows the same survey question order used in your thesis dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# APP WORKFLOW
# ============================================================
try:
    raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
except UnicodeDecodeError:
    raw_df = pd.read_csv(uploaded_file, encoding="latin1")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📄 Data Preview",
        "⚙️ Processing",
        "📊 Model Results",
        "🧠 XAI Explanation",
        "🛠️ Security Improvement",
        "⬇️ Downloads",
    ]
)

# ------------------------
# DATA PREVIEW TAB
# ------------------------
with tab1:
    section_header(
        "📄 Raw Uploaded Data",
        "Preview of the uploaded survey dataset before processing.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        custom_metric("Total Rows", raw_df.shape[0])
    with col2:
        custom_metric("Total Columns", raw_df.shape[1])


    st.markdown("### Raw Data Sample")
    st.dataframe(raw_df.head(10), use_container_width=True)

# ------------------------
# PROCESSING
# ------------------------
clean_df = prepare_raw_csv(raw_df)
encoded_df = encode_survey(clean_df)
scored_df, scoring_info = create_risk_scores(encoded_df)
ml_dataset = create_ml_dataset(scored_df)

with tab2:
    section_header(
        "⚙️ Data Cleaning, Encoding, and Risk Scoring",
        "The system converts text answers into numeric cybersecurity behavior scores.",
    )

    st.markdown(
        '<div class="info-box">1 = risky behavior, 2 = moderate behavior, 3 = safer behavior.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Cleaned Dataset")
    st.dataframe(clean_df.head(10), use_container_width=True)

    st.markdown("### Encoded Dataset")
    st.dataframe(encoded_df.head(10), use_container_width=True)

    st.markdown("### Scoring Information")
    col1, col2 = st.columns(2)
    with col1:
        custom_metric("Scoring Questions", len(scoring_info["active_score_columns"]))
    with col2:
        custom_metric("Minimum Score", scoring_info["min_score"])
    with col3:
        custom_metric("Maximum Score", scoring_info["max_score"])

    st.write("Active scoring columns:")
    st.write(scoring_info["active_score_columns"])

    st.markdown("### Risk Score Preview")
    st.dataframe(
        scored_df[
            ["Cyber_Safety_Score", "Cyber_Risk_Score", "Risk_Level", "Risk_Level_Numeric"]
        ].head(10),
        use_container_width=True,
    )

    st.markdown("### Risk Distribution")
    risk_counts = scored_df["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        text="Count",
        title="Risk Level Distribution",
        color="Risk Level",
        color_discrete_map={
            "Low Risk": "#16a34a",
            "Medium Risk": "#ca8a04",
            "High Risk": "#dc2626",
        },
    )
    fig_risk.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("### Final ML Dataset")
    st.dataframe(ml_dataset.head(10), use_container_width=True)


# ------------------------
# MODEL TRAINING
# ------------------------
lr_model, scaler, X_train, X_test, y_train, y_test, y_pred, result = train_logistic_regression(ml_dataset)

with tab3:
    section_header(
        "📊 Logistic Regression Model Performance",
        "Evaluation of the trained cybersecurity risk prediction model.",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        custom_metric("Accuracy", f"{result['Accuracy']:.4f}")
    with col2:
        custom_metric("Precision", f"{result['Precision']:.4f}")
    with col3:
        custom_metric("Recall", f"{result['Recall']:.4f}")
    with col4:
        custom_metric("F1-score", f"{result['F1-score']:.4f}")

    st.markdown("### Train/Test Size")
    c1, c2 = st.columns(2)
    with c1:
        custom_metric("Training Rows", X_train.shape[0])
    with c2:
        custom_metric("Testing Rows", X_test.shape[0])

    st.markdown("### Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low Risk", "Medium Risk", "High Risk"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Predicted Low", "Predicted Medium", "Predicted High"],
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig_cm.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.dataframe(cm_df, use_container_width=True)


# ------------------------
# XAI TAB
# ------------------------
with tab4:
    section_header(
        "🧠 Explainable AI: SHAP and LIME",
        "Explain which cybersecurity behaviors influence the risk prediction.",
    )

    feature_names = X_train.columns.tolist()

    def app_predict_proba(data):
        return predict_proba_original(data, lr_model, scaler, feature_names)

    st.markdown("### SHAP Global Explanation")
    st.write(
        "SHAP shows the most important features influencing a selected risk class. "
        "For thesis explanation, High Risk is usually the most important class."
    )

    available_classes = list(lr_model.classes_)
    selected_class = st.selectbox(
        "Select class to explain",
        available_classes,
        format_func=lambda x: class_names.get(x, str(x)),
        index=available_classes.index(2) if 2 in available_classes else 0,
    )

    if st.button("Generate SHAP Explanation"):
        with st.spinner("Generating SHAP explanation..."):
            background_size = min(40, len(X_train))
            shap_sample_size = min(30, len(X_test))

            background = shap.sample(X_train, background_size, random_state=42)
            X_shap = X_test.sample(shap_sample_size, random_state=42)

            shap_explainer = shap.KernelExplainer(
                app_predict_proba,
                background,
            )

            shap_values = shap_explainer.shap_values(
                X_shap,
                nsamples=100,
            )

            class_position = available_classes.index(selected_class)

            if isinstance(shap_values, list):
                shap_values_selected = shap_values[class_position]
            else:
                shap_values_selected = shap_values[:, :, class_position]

            mean_abs_shap = np.abs(shap_values_selected).mean(axis=0)

            shap_importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Mean Absolute SHAP Value": mean_abs_shap,
                }
            ).sort_values(
                by="Mean Absolute SHAP Value",
                ascending=False,
            )

            st.markdown(f"#### Top SHAP Features for {class_names.get(selected_class, selected_class)}")
            st.dataframe(shap_importance_df.head(15), use_container_width=True)

            fig_shap = px.bar(
                shap_importance_df.head(15).sort_values("Mean Absolute SHAP Value"),
                x="Mean Absolute SHAP Value",
                y="Feature",
                orientation="h",
                title="Top 15 SHAP Feature Importance",
                color="Mean Absolute SHAP Value",
                color_continuous_scale="Teal",
            )
            fig_shap.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis_title="Feature",
                xaxis_title="Mean Absolute SHAP Value",
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")
    st.markdown("### LIME Local Explanation")
    st.write("LIME explains one individual student's prediction.")

    if len(X_test) > 0:
        student_index = st.number_input(
            "Select test student index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1,
        )

        if st.button("Generate LIME Explanation"):
            with st.spinner("Generating LIME explanation..."):
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train.values,
                    feature_names=feature_names,
                    class_names=["Low Risk", "Medium Risk", "High Risk"],
                    mode="classification",
                    discretize_continuous=True,
                    random_state=42,
                )

                selected_student = X_test.iloc[int(student_index)].values

                lime_exp = lime_explainer.explain_instance(
                    data_row=selected_student,
                    predict_fn=app_predict_proba,
                    num_features=10,
                    top_labels=3,
                )

                predicted_class = lr_model.predict(
                    scaler.transform(X_test.iloc[[int(student_index)]])
                )[0]

                st.markdown(
                    f"Predicted Risk: {risk_badge(class_names.get(predicted_class, predicted_class))}",
                    unsafe_allow_html=True,
                )

                st.markdown("#### Top LIME Reasons")
                lime_list = lime_exp.as_list(label=predicted_class)
                lime_df = pd.DataFrame(lime_list, columns=["Feature Rule", "Contribution"])
                st.dataframe(lime_df, use_container_width=True)

                components.html(lime_exp.as_html(), height=750, scrolling=True)



# ------------------------
# RECOMMENDATIONS TAB
# ------------------------
with tab5:
    section_header(
        "🛠️ Security Improvement Recommendations",
        "Actionable suggestions based on the weakest cybersecurity behavior areas in the uploaded survey data.",
    )

    st.markdown(
        """
        <div class="info-box">
            This section converts model and scoring results into practical cybersecurity improvement actions.
            Lower average scores indicate weaker security behavior and higher improvement priority.
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommendation_df = build_recommendation_table(scored_df)

    if recommendation_df.empty:
        st.warning("No recommendation data could be generated.")
    else:
        st.markdown("### Overall Weak Areas")
        st.dataframe(recommendation_df.head(10), use_container_width=True)

        fig_rec = px.bar(
            recommendation_df.head(10).sort_values("Average Score"),
            x="Average Score",
            y="Topic",
            orientation="h",
            title="Top Cybersecurity Areas Needing Improvement",
            color="Weak/Moderate %",
            color_continuous_scale="Reds",
        )
        fig_rec.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis_title="Improvement Area",
            xaxis_title="Average Security Score",
        )
        st.plotly_chart(fig_rec, use_container_width=True)

        st.markdown("### Recommended Institutional Actions")

        for _, row in recommendation_df.head(6).iterrows():
            rec = recommendation_bank[row["Feature"]]
            priority = "High Priority" if row["Average Score"] <= 1.75 else "Medium Priority"

            item = {
                "Priority": priority,
                "Topic": rec["title"],
                "Issue": rec["issue"],
                "Recommended Action": rec["action"],
                "Details": rec["details"],
            }

            st.markdown(recommendation_card(item), unsafe_allow_html=True)

        st.markdown("### Risk-Based Improvement Plan")

        high_risk_count = int((scored_df["Risk_Level"] == "High Risk").sum())
        medium_risk_count = int((scored_df["Risk_Level"] == "Medium Risk").sum())
        low_risk_count = int((scored_df["Risk_Level"] == "Low Risk").sum())

        col1, col2, col3 = st.columns(3)
        with col1:
            custom_metric("High Risk Students", high_risk_count)
        with col2:
            custom_metric("Medium Risk Students", medium_risk_count)
        with col3:
            custom_metric("Low Risk Students", low_risk_count)

        st.markdown(
            """
            <div class="rec-card">
                <div class="rec-title">Immediate Actions</div>
                <div class="rec-text">
                    Focus on High Risk students first. Arrange awareness sessions on password reuse,
                    phishing links, 2FA, password managers, and cracked software.
                </div>
            </div>
            <div class="rec-card">
                <div class="rec-title">Short-Term Actions</div>
                <div class="rec-text">
                    Conduct department-wise cybersecurity workshops and share short monthly learning content.
                    Use posters, email reminders, and short quizzes to improve awareness.
                </div>
            </div>
            <div class="rec-card">
                <div class="rec-title">Long-Term Actions</div>
                <div class="rec-text">
                    Repeat the survey every semester and compare risk distribution over time.
                    Use the dashboard to track whether awareness programs reduce High Risk behavior.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Individual Student Improvement Suggestions")

        student_row_index = st.number_input(
            "Select row number from processed dataset",
            min_value=0,
            max_value=len(scored_df) - 1,
            value=0,
            step=1,
        )

        selected_student_row = scored_df.iloc[int(student_row_index)]
        selected_risk = selected_student_row["Risk_Level"]

        st.markdown(
            f"Selected Student Risk Level: {risk_badge(selected_risk)}",
            unsafe_allow_html=True,
        )

        individual_items = get_individual_recommendations(selected_student_row, max_items=6)

        if len(individual_items) == 0:
            st.success("This student has no major weak cybersecurity behavior based on the available scoring items.")
        else:
            for item in individual_items:
                st.markdown(recommendation_card(item), unsafe_allow_html=True)


# ------------------------
# DOWNLOADS TAB
# ------------------------
with tab6:
    section_header(
        "⬇️ Download Outputs",
        "Download processed files for thesis documentation and further analysis.",
    )

    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.download_button(
        label="Download Processed Risk Dataset",
        data=to_csv_bytes(scored_df),
        file_name="processed_risk_dataset.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.download_button(
        label="Download Final ML Dataset",
        data=to_csv_bytes(ml_dataset),
        file_name="final_ml_dataset.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.download_button(
        label="Download Classification Report",
        data=to_csv_bytes(report_df.reset_index().rename(columns={"index": "Metric"})),
        file_name="classification_report.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.download_button(
        label="Download Confusion Matrix",
        data=to_csv_bytes(cm_df.reset_index().rename(columns={"index": "Actual Class"})),
        file_name="confusion_matrix.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    recommendation_df_for_download = build_recommendation_table(scored_df)

    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.download_button(
        label="Download Security Recommendations",
        data=to_csv_bytes(recommendation_df_for_download),
        file_name="security_recommendations.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)
