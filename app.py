import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

# Import prediction engine from backend API gateway
from src.api_gateway import predict_flight_risk, load_model_bundle, MODEL_PATH

# ---------------------------------------------------------
# Page Configuration & Global Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="R.E.T.A.I.N. — Predictive HR Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        color: #f8fafc;
    }
    
    /* Risk Badges */
    .risk-badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    /* Stressor Pill */
    .stressor-pill {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 6px 4px 0;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Preset Profiles Definition
# ---------------------------------------------------------
PRESETS = {
    "Chronic Overtime Burnout (1-Year Overtime & WLB 1)": {
        "Age": 32, "BusinessTravel": "Travel_Frequently", "DailyRate": 400, "Department": "Sales",
        "DistanceFromHome": 15, "Education": 3, "EducationField": "Marketing", "EnvironmentSatisfaction": 2,
        "Gender": "Male", "HourlyRate": 45, "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Sales Executive",
        "JobSatisfaction": 2, "MaritalStatus": "Single", "MonthlyIncome": 3200, "MonthlyRate": 14000,
        "NumCompaniesWorked": 4, "OverTime": "Yes", "PercentSalaryHike": 11, "PerformanceRating": 3,
        "RelationshipSatisfaction": 2, "StockOptionLevel": 0, "TotalWorkingYears": 6, "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 1, "YearsAtCompany": 2, "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 1
    },
    "Severe Wage Deprivation ($100 - $500 Low Salary)": {
        "Age": 45, "BusinessTravel": "Non-Travel", "DailyRate": 1200, "Department": "Research & Development",
        "DistanceFromHome": 10, "Education": 4, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 3,
        "Gender": "Female", "HourlyRate": 120, "JobInvolvement": 4, "JobLevel": 4, "JobRole": "Manager",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 100, "MonthlyRate": 1000,
        "NumCompaniesWorked": 8, "OverTime": "No", "PercentSalaryHike": 18, "PerformanceRating": 4,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 2, "TotalWorkingYears": 15, "TrainingTimesLastYear": 4,
        "WorkLifeBalance": 3, "YearsAtCompany": 8, "YearsInCurrentRole": 6, "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 5
    },
    "Extreme Commute Outlier (100km Distance, All Else Safe)": {
        "Age": 40, "BusinessTravel": "Non-Travel", "DailyRate": 1400, "Department": "Research & Development",
        "DistanceFromHome": 100, "Education": 4, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 4,
        "Gender": "Female", "HourlyRate": 95, "JobInvolvement": 4, "JobLevel": 4, "JobRole": "Manager",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 15000, "MonthlyRate": 20000,
        "NumCompaniesWorked": 1, "OverTime": "No", "PercentSalaryHike": 20, "PerformanceRating": 4,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 3, "TotalWorkingYears": 18, "TrainingTimesLastYear": 4,
        "WorkLifeBalance": 4, "YearsAtCompany": 10, "YearsInCurrentRole": 8, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 8
    },
    "Safe Retention (Competitive Salary & High Engagement)": {
        "Age": 42, "BusinessTravel": "Non-Travel", "DailyRate": 1200, "Department": "Research & Development",
        "DistanceFromHome": 5, "Education": 4, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 4,
        "Gender": "Female", "HourlyRate": 85, "JobInvolvement": 4, "JobLevel": 3, "JobRole": "Research Director",
        "JobSatisfaction": 4, "MaritalStatus": "Married", "MonthlyIncome": 13500, "MonthlyRate": 21000,
        "NumCompaniesWorked": 1, "OverTime": "No", "PercentSalaryHike": 19, "PerformanceRating": 4,
        "RelationshipSatisfaction": 4, "StockOptionLevel": 2, "TotalWorkingYears": 20, "TrainingTimesLastYear": 4,
        "WorkLifeBalance": 4, "YearsAtCompany": 12, "YearsInCurrentRole": 8, "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 8
    }
}


# Initialize Session State
if "employee_data" not in st.session_state:
    st.session_state.employee_data = PRESETS["Chronic Overtime Burnout (1-Year Overtime & WLB 1)"].copy()


def load_preset(preset_name: str):
    st.session_state.employee_data = PRESETS[preset_name].copy()


# Ensure model is ready
@st.cache_resource
def get_cached_model():
    return load_model_bundle()

try:
    model_bundle = get_cached_model()
    model_status_badge = "🟢 Model Active (Random Forest v1.0 + Boundary Engine)"
except Exception as e:
    model_status_badge = f"🔴 Model Error: {e}"


# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🛡️ R.E.T.A.I.N. — ML Engine")
    st.caption("**Risk Evaluation Tool for Attrition Insights & Navigation** | Enterprise Flight Risk & Boundary Condition Engine")

with col_head2:
    st.markdown(f"<div style='text-align: right; padding-top: 15px;'><code>{model_status_badge}</code></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "👤 Single Employee Evaluator",
    "📊 Batch CSV Analytics & Upload",
    "🧠 Model Explainability & Insights"
])


# =========================================================
# TAB 1: Single Employee Evaluator
# =========================================================
with tab1:
    st.markdown("### 🎯 Interactive Employee Risk Evaluation")
    st.write("Adjust behavioral and demographic indicators below or choose a preset to evaluate attrition probability, boundary dealbreakers, and primary stressors in real time.")

    # Preset Profile Selector Bar
    st.markdown("##### ⚡ Quick Preset Profiles & Boundary Condition Scenarios")
    preset_cols = st.columns(4)
    if preset_cols[0].button("🛑 1-Yr Overtime Burnout", use_container_width=True):
        load_preset("Chronic Overtime Burnout (1-Year Overtime & WLB 1)")
        st.rerun()
    if preset_cols[1].button("🛑 Extreme Low Salary ($100)", use_container_width=True):
        load_preset("Severe Wage Deprivation ($100 - $500 Low Salary)")
        st.rerun()
    if preset_cols[2].button("🛑 100km Commute Infeasibility", use_container_width=True):
        load_preset("Extreme Commute Outlier (100km Distance, All Else Safe)")
        st.rerun()
    if preset_cols[3].button("🟢 Safe Retention Profile", use_container_width=True):
        load_preset("Safe Retention (Competitive Salary & High Engagement)")
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    curr = st.session_state.employee_data

    # Input Form Layout (2 Columns)
    col_inputs, col_results = st.columns([7, 5], gap="large")

    with col_inputs:
        with st.expander("👤 1. Demographics & Commute", expanded=True):
            d_col1, d_col2 = st.columns(2)
            age = d_col1.slider("Age (Years)", min_value=18, max_value=65, value=int(curr.get("Age", 30)))
            gender = d_col2.selectbox("Gender", ["Male", "Female"], index=0 if curr.get("Gender") == "Male" else 1)
            marital_status = d_col1.selectbox("Marital Status", ["Single", "Married", "Divorced"], 
                                               index=["Single", "Married", "Divorced"].index(curr.get("MaritalStatus", "Single")))
            distance = d_col2.number_input("Distance From Home (km)", min_value=1, max_value=200, value=int(curr.get("DistanceFromHome", 10)))
            
            edu_col1, edu_col2 = st.columns(2)
            education_levels = {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}
            education = edu_col1.selectbox("Education Level", options=[1, 2, 3, 4, 5], 
                                           format_func=lambda x: education_levels[x], 
                                           index=min(4, max(0, int(curr.get("Education", 3)) - 1)))
            education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
            education_field = edu_col2.selectbox("Education Field", education_fields, 
                                                 index=education_fields.index(curr.get("EducationField", "Life Sciences")) if curr.get("EducationField") in education_fields else 0)

        with st.expander("💼 2. Role & Compensation", expanded=True):
            r_col1, r_col2 = st.columns(2)
            departments = ["Sales", "Research & Development", "Human Resources"]
            department = r_col1.selectbox("Department", departments, index=departments.index(curr.get("Department", "Sales")))
            
            roles = [
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager", 
                "Sales Representative", "Research Director", "Human Resources"
            ]
            job_role = r_col2.selectbox("Job Role", roles, index=roles.index(curr.get("JobRole", "Sales Executive")) if curr.get("JobRole") in roles else 0)
            
            job_level = r_col1.selectbox("Job Level (1 to 5)", [1, 2, 3, 4, 5], index=int(curr.get("JobLevel", 1)) - 1)
            monthly_income = r_col2.number_input("Monthly Income ($ / Rs)", min_value=50, max_value=50000, value=int(curr.get("MonthlyIncome", 4000)), step=100)
            
            c_col1, c_col2 = st.columns(2)
            salary_hike = c_col1.slider("Percent Salary Hike (%)", min_value=10, max_value=25, value=int(curr.get("PercentSalaryHike", 14)))
            stock_option = c_col2.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3], index=int(curr.get("StockOptionLevel", 0)))


        with st.expander("🧠 3. Workplace Sentiment & Burnout Ratings", expanded=True):
            s_col1, s_col2 = st.columns(2)
            satisfaction_labels = {1: "1 - Low / Dissatisfied", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
            job_sat = s_col1.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=lambda x: satisfaction_labels[x], index=int(curr.get("JobSatisfaction", 3)) - 1)
            env_sat = s_col2.selectbox("Environment Satisfaction", [1, 2, 3, 4], format_func=lambda x: satisfaction_labels[x], index=int(curr.get("EnvironmentSatisfaction", 3)) - 1)
            rel_sat = s_col1.selectbox("Relationship Satisfaction", [1, 2, 3, 4], format_func=lambda x: satisfaction_labels[x], index=int(curr.get("RelationshipSatisfaction", 3)) - 1)
            
            wlb_labels = {1: "1 - Bad (Severe Burnout)", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
            work_life = s_col2.selectbox("Work-Life Balance", [1, 2, 3, 4], format_func=lambda x: wlb_labels[x], index=int(curr.get("WorkLifeBalance", 3)) - 1)
            job_inv = s_col1.selectbox("Job Involvement", [1, 2, 3, 4], format_func=lambda x: satisfaction_labels[x], index=int(curr.get("JobInvolvement", 3)) - 1)
            perf_rating = s_col2.selectbox("Performance Rating", [3, 4], format_func=lambda x: f"{x} - {'Excellent' if x==3 else 'Outstanding'}", index=0 if curr.get("PerformanceRating", 3) == 3 else 1)

        with st.expander("⏳ 4. Career History & Working Patterns", expanded=True):
            h_col1, h_col2 = st.columns(2)
            overtime = h_col1.radio("Mandatory Overtime", ["Yes", "No"], index=0 if curr.get("OverTime") == "Yes" else 1, horizontal=True)
            travel_options = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
            bus_travel = h_col2.selectbox("Business Travel", travel_options, index=travel_options.index(curr.get("BusinessTravel", "Travel_Rarely")))
            
            total_years = h_col1.slider("Total Working Years", min_value=0, max_value=40, value=int(curr.get("TotalWorkingYears", 8)))
            years_company = h_col2.slider("Years at Current Company", min_value=0, max_value=40, value=min(int(curr.get("YearsAtCompany", 3)), total_years))
            years_role = h_col1.slider("Years in Current Role", min_value=0, max_value=30, value=min(int(curr.get("YearsInCurrentRole", 2)), years_company))
            years_promo = h_col2.slider("Years Since Last Promotion", min_value=0, max_value=20, value=min(int(curr.get("YearsSinceLastPromotion", 1)), years_company))
            years_manager = h_col1.slider("Years with Current Manager", min_value=0, max_value=20, value=min(int(curr.get("YearsWithCurrManager", 2)), years_company))
            num_companies = h_col2.number_input("Number of Companies Worked", min_value=0, max_value=15, value=int(curr.get("NumCompaniesWorked", 2)))
            training_times = h_col1.selectbox("Training Times Last Year", [0, 1, 2, 3, 4, 5, 6], index=int(curr.get("TrainingTimesLastYear", 3)))

    # Assemble Payload for Prediction
    current_payload = {
        "Age": age, "BusinessTravel": bus_travel, "DailyRate": curr.get("DailyRate", 800),
        "Department": department, "DistanceFromHome": distance, "Education": education,
        "EducationField": education_field, "EnvironmentSatisfaction": env_sat, "Gender": gender,
        "HourlyRate": curr.get("HourlyRate", 65), "JobInvolvement": job_inv, "JobLevel": job_level,
        "JobRole": job_role, "JobSatisfaction": job_sat, "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income, "MonthlyRate": curr.get("MonthlyRate", 14000),
        "NumCompaniesWorked": num_companies, "OverTime": overtime, "PercentSalaryHike": salary_hike,
        "PerformanceRating": perf_rating, "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_years, "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company, "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_manager
    }

    # Execute Prediction
    result = predict_flight_risk(current_payload)
    prob_pct = result.flight_risk_probability * 100.0

    # ---------------------------------------------------------
    # Results Display Column
    # ---------------------------------------------------------
    with col_results:
        st.markdown("### 📊 Flight Risk Assessment")

        # Visual Gauge Chart
        gauge_color = "#ef4444" if result.risk_level == "HIGH" else "#f59e0b" if result.risk_level == "MEDIUM" else "#10b981"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 42, "color": gauge_color}},
            title={"text": f"<b>{result.risk_level} RISK TIER</b>", "font": {"size": 20, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.2)",
                "steps": [
                    {"range": [0, 16], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [16, 35], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [35, 100], "color": "rgba(239, 68, 68, 0.15)"}
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.8,
                    "value": prob_pct
                }
            }
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk Summary Alert
        if result.risk_level == "HIGH":
            st.error(f"🚨 **High Flight Risk ({prob_pct:.1f}%)**: Immediate retention intervention recommended.")
        elif result.risk_level == "MEDIUM":
            st.warning(f"⚠️ **Elevated Risk ({prob_pct:.1f}%)**: Monitored employee; flight risk is higher than company baseline (16.1%).")
        else:
            st.success(f"✅ **Safe Retention ({prob_pct:.1f}%)**: Flight risk is well below company baseline.")

        # Key Workplace Stressors Callout
        st.markdown("#### 🚨 Primary Workplace Stressors")
        if result.top_stressors:
            for s in result.top_stressors:
                st.markdown(f"<span class='stressor-pill'>⚠️ {s}</span>", unsafe_allow_html=True)
        else:
            st.write("No severe active stressors detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Domain Feature Engineering Indicators
        st.markdown("#### 🔬 Engineered Ratio Diagnostics")
        r_col1, r_col2 = st.columns(2)
        
        income_per_level = monthly_income / (job_level + 1.0)
        job_hop_idx = num_companies / (total_years + 1.0)
        sat_score = (job_sat + env_sat + rel_sat + work_life) / 4.0
        tenure_ratio = years_company / (total_years + 1.0)
        
        r_col1.metric("Income per Job Level", f"${income_per_level:.0f}", 
                      delta="Underpaid" if income_per_level < 1500 else "Normal",
                      delta_color="inverse" if income_per_level < 1500 else "normal")
        r_col2.metric("Job Hopping Index", f"{job_hop_idx:.2f}",
                      delta="High Turnover" if job_hop_idx >= 0.50 else "Stable",
                      delta_color="inverse" if job_hop_idx >= 0.50 else "normal")
        r_col1.metric("Overall Sentiment Score", f"{sat_score:.2f} / 4.0",
                      delta="Burnout" if sat_score <= 2.25 else "Healthy",
                      delta_color="inverse" if sat_score <= 2.25 else "normal")
        r_col2.metric("Tenure Ratio", f"{tenure_ratio:.1%}")


# =========================================================
# TAB 2: Batch CSV Analytics & Upload
# =========================================================
with tab2:
    st.markdown("### 📊 Bulk Employee Evaluation & Department Analytics")
    st.write("Upload an employee roster CSV to score attrition probability across teams, generate risk distribution charts, and export annotated datasets.")

    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        uploaded_file = st.file_uploader("Upload Employee Data CSV", type=["csv"])
    with b_col2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("📁 Load Raw IBM Dataset Sample", use_container_width=True)

    df_batch = None
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
    elif load_sample:
        raw_path = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        if os.path.exists(raw_path):
            df_batch = pd.read_csv(raw_path).head(100)
            st.info(f"Loaded sample of 100 employee records from `{raw_path}`.")
        else:
            st.error(f"Cannot find raw dataset at `{raw_path}`.")

    if df_batch is not None:
        with st.spinner("Scoring employees through R.E.T.A.I.N. ML Engine..."):
            results_prob = []
            results_tier = []
            results_stressors = []

            for _, row in df_batch.iterrows():
                row_dict = row.to_dict()
                pred = predict_flight_risk(row_dict)
                results_prob.append(pred.flight_risk_probability)
                results_tier.append(pred.risk_level)
                results_stressors.append(", ".join(pred.top_stressors))

            df_scored = df_batch.copy()
            df_scored["Flight_Risk_Probability"] = results_prob
            df_scored["Risk_Level"] = results_tier
            df_scored["Primary_Stressors"] = results_stressors

        # Batch Summary Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        total_eval = len(df_scored)
        high_risk_count = sum(df_scored["Risk_Level"] == "HIGH")
        med_risk_count = sum(df_scored["Risk_Level"] == "MEDIUM")
        avg_risk = np.mean(df_scored["Flight_Risk_Probability"]) * 100

        m1.metric("Employees Evaluated", f"{total_eval:,}")
        m2.metric("High Risk Employees", f"{high_risk_count}", delta=f"{(high_risk_count/total_eval):.1%} of workforce", delta_color="inverse")
        m3.metric("Medium Risk Employees", f"{med_risk_count}")
        m4.metric("Average Attrition Risk", f"{avg_risk:.1f}%")

        # Interactive Visual Analytics
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            tier_counts = df_scored["Risk_Level"].value_counts().reset_index()
            tier_counts.columns = ["Risk_Level", "Count"]
            color_map = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
            fig_pie = px.pie(tier_counts, names="Risk_Level", values="Count", title="Workforce Risk Tier Breakdown",
                             color="Risk_Level", color_discrete_map=color_map, hole=0.45)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f8fafc"})
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_chart2:
            if "Department" in df_scored.columns:
                dept_risk = df_scored.groupby("Department")["Flight_Risk_Probability"].mean().reset_index()
                dept_risk["Flight_Risk_%"] = dept_risk["Flight_Risk_Probability"] * 100.0
                fig_bar = px.bar(dept_risk, x="Department", y="Flight_Risk_%", title="Average Flight Risk by Department",
                                 color="Flight_Risk_%", color_continuous_scale="Reds")
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f8fafc"})
                st.plotly_chart(fig_bar, use_container_width=True)

        # Scored Data Table & Download
        st.markdown("#### 📋 Scored Employee Records")
        st.dataframe(df_scored[["Department", "JobRole", "MonthlyIncome", "OverTime", "DistanceFromHome", "Flight_Risk_Probability", "Risk_Level", "Primary_Stressors"]], 
                     use_container_width=True)

        csv_data = df_scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Scored CSV Report",
            data=csv_data,
            file_name="retained_scored_employees.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# TAB 3: Model Explainability & Deep Dive
# =========================================================
with tab3:
    st.markdown("### 🧠 Model Explainability & Feature Importance")
    st.write("Understand the internal mechanics and top features driving Random Forest predictions in the R.E.T.A.I.N. pipeline.")

    if "feature_importances" in model_bundle:
        feat_dict = model_bundle["feature_importances"]
        df_feat = pd.DataFrame(list(feat_dict.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True).tail(15)

        fig_feat = px.bar(df_feat, x="Importance", y="Feature", orientation="h",
                           title="Top 15 Most Influential Predictive Features (Random Forest)",
                           color="Importance", color_continuous_scale="Viridis")
        fig_feat.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f8fafc"})
        st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")
    st.markdown(r"""
    #### 🔍 Key Engineering & Boundary Condition Insights:
    1. **Hard Boundary Conditions (Points-of-No-Return)**:
       - **Extreme Commute Infeasibility ($\ge 80\text{km}$)**: Irreversible fatigue trigger; floors risk at **$\ge 92\%$ (HIGH Risk)**.
       - **Severe Wage Deprivation ($<\$1,000 / \text{Low Wage}$)**: Severe economic unviability; floors risk at **$\ge 98\%$ (HIGH Risk)**.
       - **Chronic Overtime Burnout ($\text{Overtime} + \text{WLB} \le 1$)**: Acute physical/mental exhaustion; floors risk at **$\ge 90\%$ (HIGH Risk)**.
       - **Toxic Environment ($\text{EnvSat} \le 1 \ \& \ \text{JobSat} \le 1 \ \& \ \text{WLB} \le 2$)**: Complete burnout; floors risk at **$\ge 88\%$ (HIGH Risk)**.
    2. **Coupled Cross-Feature Multipliers**:
       - **Overtime $\times$ Long Commute ($\ge 35\text{km}$)**: Compounded exhaustion penalty (+25%).
       - **Overtime $\times$ Low Salary ($<\$3,500$)**: Uncompensated exploitation penalty (+20%).
       - **High Mobility $\times$ Dissatisfaction**: Rapid resignation trigger (+20%).
    3. **Baseline Calibrated Risk Tiers**:
       - Company Baseline Attrition Rate = **16.1%**.
       - `HIGH Risk Tier` ($\ge 35\%$): More than **2.18x** higher than company baseline.
       - `MEDIUM Risk Tier` ($16\% - 35\%$): Above company baseline.
       - `LOW Risk Tier` ($< 16\%$): Below company baseline.
    """)

