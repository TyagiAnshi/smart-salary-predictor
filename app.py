from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/salary_ridge.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")


def build_input_vector(data):
    row = {col: 0 for col in feature_cols}

    exp_map = {"Entry": 0, "Mid": 1, "Senior": 2, "Executive": 3}
    emp_map = {"Part-time": 0, "Freelance": 1, "Contract": 2, "Full-time": 3}
    size_map = {"Small": 0, "Medium": 1, "Large": 2}

    row["experience_level"] = exp_map.get(data.get("experience_level"), 1)
    row["employment_type"] = emp_map.get(data.get("employment_type"), 3)
    row["company_size"] = size_map.get(data.get("company_size"), 1)
    row["remote_ratio"] = int(data.get("remote_ratio", 50))
    row["work_year"] = int(data.get("work_year", 2023))

    job_col = f"job_title_{data.get('job_title', 'Data Scientist')}"
    if job_col in row:
        row[job_col] = 1

    residence_col = f"employee_residence_{data.get('residence', 'US')}"
    if residence_col in row:
        row[residence_col] = 1

    location_col = f"company_location_{data.get('company_location', 'US')}"
    if location_col in row:
        row[location_col] = 1

    return pd.DataFrame([row])


def get_ai_suggestion(data, predicted_salary):
    suggestions = []

    experience_level = data.get("experience_level")
    job_title = data.get("job_title")
    remote_ratio = int(data.get("remote_ratio", 50))
    company_size = data.get("company_size")
    employment_type = data.get("employment_type")

    if experience_level == "Entry":
        suggestions.append("Build 2-3 strong portfolio projects and add them to GitHub.")
        suggestions.append("Learn SQL, Python, machine learning basics, and data visualization.")
    elif experience_level == "Mid":
        suggestions.append("Work on advanced projects using real datasets and cloud tools.")
        suggestions.append("Improve model deployment, APIs, dashboards, and business communication skills.")
    elif experience_level == "Senior":
        suggestions.append("Focus on leadership, system design, mentoring, and end-to-end ML solutions.")
        suggestions.append("Highlight business impact, cost savings, and revenue improvements in your resume.")
    else:
        suggestions.append("Focus on strategy, team leadership, and high-level business decision making.")

    if job_title in ["Data Scientist", "ML Engineer"]:
        suggestions.append("Strengthen machine learning, deep learning, MLOps, and model deployment skills.")
    elif job_title == "Data Analyst":
        suggestions.append("Improve SQL, Excel, Power BI/Tableau, statistics, and storytelling with data.")
    elif job_title == "Data Engineer":
        suggestions.append("Learn data pipelines, Spark, cloud platforms, databases, and workflow automation.")
    else:
        suggestions.append("Build skills that match high-demand roles in data and AI.")

    if remote_ratio < 100:
        suggestions.append("Remote-friendly roles may increase opportunities in higher-paying markets.")

    if company_size == "Large":
        suggestions.append("Large companies often value specialization, certifications, and strong interview preparation.")
    else:
        suggestions.append("For smaller companies, show that you can handle multiple responsibilities independently.")

    if employment_type != "Full-time":
        suggestions.append("Full-time roles may offer more stable salary growth and benefits.")

    suggestions.append(
        f"Your predicted salary is ${predicted_salary:,.2f}. "
        "Keep improving skills and projects to increase your earning potential."
    )

    return "\n\n".join(suggestions)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = build_input_vector(data)
    predicted_salary = model.predict(input_df)[0]
    predicted_salary = max(10000, predicted_salary)

    suggestion = get_ai_suggestion(data, predicted_salary)

    return jsonify({
        "predicted_salary": round(float(predicted_salary), 2),
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(debug=True)
