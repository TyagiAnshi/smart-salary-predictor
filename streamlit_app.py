import streamlit as st
import requests
import plotly.graph_objects as go

FLASK_URL = "http://localhost:5000"

st.set_page_config(page_title="Smart Salary Predictor", page_icon="💼", layout="wide")

st.title("Smart Salary Predictor")
st.caption("Powered by Linear Regression + Claude AI suggestions")

with st.sidebar:
    st.header("Your Profile")

    job_title = st.selectbox("Job title", [
        "Data Scientist", "Data Engineer", "ML Engineer",
        "Data Analyst", "Research Scientist", "Analytics Engineer",
        "Data Architect", "BI Developer", "Other"
    ])

    experience_level = st.selectbox("Experience level", [
        "Entry", "Mid", "Senior", "Executive"
    ])

    employment_type = st.selectbox("Employment type", [
        "Full-time", "Part-time", "Contract", "Freelance"
    ])

    company_size = st.selectbox("Company size", [
        "Small", "Medium", "Large"
    ])

    remote_ratio = st.select_slider(
        "Remote ratio (%)",
        options=[0, 50, 100],
        value=50
    )

    residence = st.text_input(
        "Your country code (e.g. US, IN, GB)",
        value="US"
    ).upper()

    company_location = st.text_input(
        "Company country code",
        value="US"
    ).upper()

    work_year = st.selectbox("Work year", [2020, 2021, 2022, 2023])

    predict_btn = st.button(
        "Predict Salary",
        type="primary",
        use_container_width=True
    )

col1, col2 = st.columns([1, 1])

if predict_btn:
    payload = {
        "job_title": job_title,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_size": company_size,
        "remote_ratio": remote_ratio,
        "residence": residence,
        "company_location": company_location,
        "work_year": work_year,
    }

    with st.spinner("Predicting salary and generating AI suggestion..."):
        try:
            response = requests.post(f"{FLASK_URL}/predict", json=payload)
            response.raise_for_status()

            result = response.json()

            salary = result.get("predicted_salary")
            suggestion = result.get("suggestion", "No AI suggestion available.")

            if salary is None:
                st.error("Prediction failed. Flask did not return predicted_salary.")
            else:
                with col1:
                    st.metric("Predicted Salary", f"${salary:,.2f}")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=salary,
                        title={"text": "Salary Prediction"},
                        gauge={
                            "axis": {
                                "range": [0, max(300000, salary * 1.2)]
                            }
                        }
                    ))

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("AI Career Suggestion")
                    st.write(suggestion)

        except requests.exceptions.ConnectionError:
            st.error("Flask backend is not running. Start app.py first.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
