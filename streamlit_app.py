from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai

from monitor import log_prediction, check_drift

# -------------------------------
# CONFIG
# -------------------------------
MODEL_VERSION = "v1.0"

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Lead Scoring.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model" / "xgboost_model.pkl"
LOGS_PATH = BASE_DIR / "logs.json"

st.set_page_config(
    page_title="Lead Prioritisation CRM Tool",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# GEMINI SETUP
# -------------------------------
GEMINI_AVAILABLE = False

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("models/gemini-flash-lite-latest")
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -------------------------------
# PATH CHECK
# -------------------------------
def check_required_file(path, label):
    if not path.exists():
        st.error(f"{label} not found at: {path}")
        st.stop()


check_required_file(DATA_PATH, "Dataset")
check_required_file(MODEL_PATH, "Model file")

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f6f8fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
}

.dark-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 24px;
    border-radius: 18px;
    color: white;
    box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.15);
    margin-bottom: 18px;
}

.export-card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0px 8px 22px rgba(15, 23, 42, 0.08);
    margin-bottom: 18px;
    border-left: 8px solid #1e3a8a;
}

.high {
    border-left: 8px solid #16a34a;
}

.medium {
    border-left: 8px solid #f59e0b;
}

.low {
    border-left: 8px solid #dc2626;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background-color: #e0f2fe;
    color: #075985;
    font-size: 13px;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}

.badge-green {
    background-color: #dcfce7;
    color: #166534;
}

.badge-yellow {
    background-color: #fef3c7;
    color: #92400e;
}

.badge-red {
    background-color: #fee2e2;
    color: #991b1b;
}

.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 8px;
}

.small-text {
    font-size: 13px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "Prospect ID",
        "Lead Source",
        "Do Not Email",
        "Do Not Call",
        "TotalVisits",
        "Total Time Spent on Website",
        "Page Views Per Visit",
        "Converted"
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in dataset: {missing_columns}")
        st.stop()

    data = data[required_columns].dropna().copy()

    data["EngagementScore"] = (
        data["TotalVisits"] * data["Page Views Per Visit"]
    )

    data["TimePerVisit"] = (
        data["Total Time Spent on Website"] / (data["TotalVisits"] + 1)
    )

    return data


df = load_data()

MODEL_FEATURES = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "EngagementScore",
    "TimePerVisit"
]

# -------------------------------
# LOGIC FUNCTIONS
# -------------------------------
def assign_segment(engagement_score, time_per_visit):
    if engagement_score >= 20:
        return "High Engagement Segment"
    elif time_per_visit >= 150 or engagement_score >= 5:
        return "Medium Engagement Segment"
    else:
        return "Low Engagement Segment"


def assign_priority(probability):
    if probability >= 0.70:
        return "High Priority"
    elif probability >= 0.40:
        return "Medium Priority"
    else:
        return "Low Priority"


def get_contact_status(do_not_email, do_not_call):
    if do_not_email == "Yes" and do_not_call == "Yes":
        return "No direct contact allowed"
    elif do_not_email == "Yes":
        return "Email restricted"
    elif do_not_call == "Yes":
        return "Phone restricted"
    else:
        return "Direct contact allowed"


def assign_recommendation_category(priority, contact_status):
    if priority == "High Priority" and contact_status == "Direct contact allowed":
        return "Immediate Sales Follow-up"
    elif priority == "High Priority" and contact_status != "Direct contact allowed":
        return "Compliant Alternative Follow-up"
    elif priority == "Medium Priority":
        return "Targeted Nurturing"
    else:
        return "Automated Low-Intensity Marketing"


def generate_recommendation(recommendation_category):
    if recommendation_category == "Immediate Sales Follow-up":
        return "Contact lead immediately through sales outreach."
    elif recommendation_category == "Targeted Nurturing":
        return "Place lead in targeted nurturing campaign."
    elif recommendation_category == "Compliant Alternative Follow-up":
        return "Use compliant alternative communication strategy."
    else:
        return "Use automated low-intensity marketing workflow."


def human_review_status(priority, contact_status):
    if priority == "High Priority":
        return "Required"
    elif contact_status != "Direct contact allowed":
        return "Required"
    else:
        return "Not Required"


def human_review_label(priority, contact_status):
    if contact_status != "Direct contact allowed":
        return "Requires compliance review"
    elif priority == "High Priority":
        return "Requires marketing/sales approval"
    else:
        return "Can be processed automatically"


def gemini_recommendation(
    priority,
    source,
    probability,
    contact_status,
    recommendation_category,
    lead_segment
):
    fallback = generate_recommendation(recommendation_category)

    if not GEMINI_AVAILABLE:
        return fallback

    prompt = f"""
You are a CRM decision-support agent.

Lead information:
- Lead source: {source}
- Predicted conversion probability: {probability:.1%}
- Priority: {priority}
- Engagement segment: {lead_segment}
- Contact status: {contact_status}
- Recommendation category: {recommendation_category}

Write one short practical recommendation for a marketing or sales team.
Respect contact restrictions.
Keep it simple, professional, and business-oriented.
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return fallback


@st.cache_data
def build_export_data(data):
    export_df = data.copy()

    probabilities = model.predict_proba(export_df[MODEL_FEATURES])[:, 1]

    export_df["PredictedProbability"] = probabilities
    export_df["LeadScore"] = (export_df["PredictedProbability"] * 100).round(0).astype(int)
    export_df["Priority"] = export_df["PredictedProbability"].apply(assign_priority)

    export_df["LeadSegment"] = export_df.apply(
        lambda row: assign_segment(row["EngagementScore"], row["TimePerVisit"]),
        axis=1
    )

    export_df["ContactStatus"] = export_df.apply(
        lambda row: get_contact_status(row["Do Not Email"], row["Do Not Call"]),
        axis=1
    )

    export_df["RecommendationCategory"] = export_df.apply(
        lambda row: assign_recommendation_category(
            row["Priority"],
            row["ContactStatus"]
        ),
        axis=1
    )

    export_df["AIRecommendation"] = export_df["RecommendationCategory"].apply(
        generate_recommendation
    )

    export_df["HumanReviewStatus"] = export_df.apply(
        lambda row: human_review_label(row["Priority"], row["ContactStatus"]),
        axis=1
    )

    return export_df


export_df = build_export_data(df)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("CRM Control Panel")
st.sidebar.caption(f"Model version: {MODEL_VERSION}")

page = st.sidebar.radio(
    "Navigation",
    [
        "Lead Evaluation",
        "Lead Export",
        "Monitoring Dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Pipeline")
st.sidebar.write("Lead data → XGBoost → AI recommendation → Human review")

if GEMINI_AVAILABLE:
    st.sidebar.success("Gemini agent active")
else:
    st.sidebar.warning("Gemini unavailable: fallback logic active")

# -------------------------------
# HERO
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>Lead Prioritisation in CRM Systems</h1>
    <p>AI-assisted CRM prototype for predicting conversion potential, assigning lead priority, and supporting marketing/sales decisions.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PAGE 1: LEAD EVALUATION
# =========================================================
if page == "Lead Evaluation":

    selected_index = st.sidebar.selectbox(
        "Select lead from dataset",
        df.index
    )

    selected = df.loc[selected_index]

    st.sidebar.markdown("---")
    st.sidebar.write("**Lead Source:**", selected["Lead Source"])
    st.sidebar.write("**Do Not Email:**", selected["Do Not Email"])
    st.sidebar.write("**Do Not Call:**", selected["Do Not Call"])

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Lead Behaviour Input")

        total_visits = st.number_input(
            "Total Visits",
            value=int(selected["TotalVisits"]),
            step=1,
            format="%d"
        )

        time_spent = st.number_input(
            "Total Time Spent on Website",
            value=int(selected["Total Time Spent on Website"]),
            step=1,
            format="%d"
        )

        page_views = st.number_input(
            "Page Views Per Visit",
            value=float(selected["Page Views Per Visit"]),
            step=0.1,
            format="%.2f"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        engagement_score = total_visits * page_views
        time_per_visit = time_spent / (total_visits + 1)

        lead_segment = assign_segment(engagement_score, time_per_visit)

        contact_status = get_contact_status(
            selected["Do Not Email"],
            selected["Do Not Call"]
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Lead Context")

        st.write(f"**Prospect ID:** `{selected['Prospect ID']}`")
        st.write(f"**Lead Source:** {selected['Lead Source']}")
        st.write(f"**Do Not Email:** {selected['Do Not Email']}")
        st.write(f"**Do Not Call:** {selected['Do Not Call']}")

        st.markdown("---")

        if lead_segment == "High Engagement Segment":
            st.markdown(
                '<span class="badge badge-green">High Engagement Segment</span>',
                unsafe_allow_html=True
            )
        elif lead_segment == "Medium Engagement Segment":
            st.markdown(
                '<span class="badge badge-yellow">Medium Engagement Segment</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="badge badge-red">Low Engagement Segment</span>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<span class="badge">Source: {selected["Lead Source"]}</span>',
            unsafe_allow_html=True
        )

        if contact_status != "Direct contact allowed":
            st.markdown(
                f'<span class="badge badge-red">{contact_status}</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="badge badge-green">Direct Contact Allowed</span>',
                unsafe_allow_html=True
            )

        st.write(f"**Engagement Score:** {engagement_score:.2f}")
        st.write(f"**Time Per Visit:** {time_per_visit:.2f}")

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Evaluate Lead", use_container_width=True):

        features = pd.DataFrame([{
            "TotalVisits": total_visits,
            "Total Time Spent on Website": time_spent,
            "Page Views Per Visit": page_views,
            "EngagementScore": engagement_score,
            "TimePerVisit": time_per_visit
        }])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        score = int(probability * 100)
        priority = assign_priority(probability)

        recommendation_category = assign_recommendation_category(
            priority,
            contact_status
        )

        recommendation = gemini_recommendation(
            priority,
            selected["Lead Source"],
            probability,
            contact_status,
            recommendation_category,
            lead_segment
        )

        human_review = human_review_status(
            priority,
            contact_status
        )

        st.markdown("---")
        st.subheader("CRM Decision Output")

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)

        with result_col1:
            st.metric("Lead Score", f"{score}/100")

        with result_col2:
            st.metric("Conversion Probability", f"{probability:.2%}")

        with result_col3:
            if priority == "High Priority":
                st.success(priority)
            elif priority == "Medium Priority":
                st.warning(priority)
            else:
                st.error(priority)

        with result_col4:
            st.info(recommendation_category)

        review_col1, review_col2 = st.columns(2)

        with review_col1:
            st.markdown("""
            <div class="dark-card">
                <h3>AI Recommendation</h3>
            """, unsafe_allow_html=True)

            st.markdown(
                f"<p style='font-size:18px;'>{recommendation}</p>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with review_col2:
            st.markdown("""
            <div class="dark-card">
                <h3>Human Approval</h3>
            """, unsafe_allow_html=True)

            if human_review == "Required":
                st.warning("Marketing or sales approval required before action.")
            else:
                st.success("Can be processed automatically.")

            st.markdown("</div>", unsafe_allow_html=True)

        log_prediction(features, prediction, probability)

        drift = check_drift(features)
        if drift:
            st.warning(f"⚠️ Potential drift detected: {', '.join(drift)}")

# =========================================================
# PAGE 2: LEAD EXPORT
# =========================================================
elif page == "Lead Export":

    st.markdown('<div class="section-title">Lead Export Dashboard</div>', unsafe_allow_html=True)
    st.caption(
        "Filter leads by priority, engagement segment, contact status, recommendation category, and lead source. "
        "Download the selected leads as a CSV file for CRM follow-up."
    )

    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Total Leads", f"{len(export_df):,}")

    with k2:
        st.metric("High Priority", f"{(export_df['Priority'] == 'High Priority').sum():,}")

    with k3:
        st.metric("Direct Contact Allowed", f"{(export_df['ContactStatus'] == 'Direct contact allowed').sum():,}")

    with k4:
        st.metric(
            "Needs Review",
            f"{(export_df['HumanReviewStatus'] != 'Can be processed automatically').sum():,}"
        )

    st.markdown("---")

    st.markdown('<div class="export-card">', unsafe_allow_html=True)
    st.subheader("Filter Lead List")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        priority_filter = st.multiselect(
            "Priority",
            options=sorted(export_df["Priority"].unique()),
            default=list(export_df["Priority"].unique())
        )

        segment_filter = st.multiselect(
            "Engagement Segment",
            options=sorted(export_df["LeadSegment"].unique()),
            default=list(export_df["LeadSegment"].unique())
        )

        contact_filter = st.multiselect(
            "Contact Status",
            options=sorted(export_df["ContactStatus"].unique()),
            default=list(export_df["ContactStatus"].unique())
        )

    with filter_col2:
        recommendation_filter = st.multiselect(
            "Recommendation Category",
            options=sorted(export_df["RecommendationCategory"].unique()),
            default=list(export_df["RecommendationCategory"].unique())
        )

        review_filter = st.multiselect(
            "Human Review Status",
            options=sorted(export_df["HumanReviewStatus"].unique()),
            default=list(export_df["HumanReviewStatus"].unique())
        )

        source_filter = st.multiselect(
            "Lead Source",
            options=sorted(export_df["Lead Source"].unique()),
            default=list(export_df["Lead Source"].unique())
        )

    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = export_df[
        (export_df["Priority"].isin(priority_filter)) &
        (export_df["LeadSegment"].isin(segment_filter)) &
        (export_df["ContactStatus"].isin(contact_filter)) &
        (export_df["RecommendationCategory"].isin(recommendation_filter)) &
        (export_df["HumanReviewStatus"].isin(review_filter)) &
        (export_df["Lead Source"].isin(source_filter))
    ].copy()

    filtered_df = filtered_df.sort_values(
        "PredictedProbability",
        ascending=False
    )

    st.markdown("---")
    st.subheader("Filtered Lead List")
    st.caption(f"{len(filtered_df):,} leads match the selected filters.")

    display_cols = [
        "Prospect ID",
        "Lead Source",
        "LeadSegment",
        "LeadScore",
        "PredictedProbability",
        "Priority",
        "ContactStatus",
        "RecommendationCategory",
        "HumanReviewStatus",
        "AIRecommendation",
        "Do Not Email",
        "Do Not Call"
    ]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=470
    )

    csv = filtered_df[display_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Lead List",
        data=csv,
        file_name="filtered_crm_leads.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# PAGE 3: MONITORING DASHBOARD
# =========================================================
elif page == "Monitoring Dashboard":

    st.markdown('<div class="section-title">Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.caption("Tracks prediction activity and basic model behaviour over time.")

    st.markdown("---")

    if LOGS_PATH.exists() and LOGS_PATH.stat().st_size > 0:

        logs = pd.read_json(LOGS_PATH, lines=True)

        m1, m2 = st.columns(2)

        with m1:
            st.metric("Total Predictions", len(logs))

        with m2:
            if "probability" in logs.columns:
                st.metric("Average Probability", f"{logs['probability'].mean():.2%}")

        chart1, chart2 = st.columns(2)

        with chart1:
            if "probability" in logs.columns:
                st.subheader("Prediction Probability Over Time")
                st.line_chart(logs["probability"])

        with chart2:
            if "prediction" in logs.columns:
                st.subheader("Prediction Class Distribution")
                st.bar_chart(logs["prediction"].value_counts())

    else:
        st.info("No predictions logged yet.")
