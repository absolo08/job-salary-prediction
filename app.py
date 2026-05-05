from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# load trained pipeline
model = joblib.load("excellent_salary_prediction.pkl")


def exp_bucket(x):
    if x <= 2:
        return "entry"
    elif x <= 5:
        return "junior"
    elif x <= 10:
        return "mid"
    elif x <= 15:
        return "senior"
    return "lead"


education_map = {
    "High School": 0,
    "Diploma": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4,
}

company_map = {
    "Startup": 0,
    "Small": 1,
    "Medium": 2,
    "Large": 3,
    "Enterprise": 4,
}

remote_map = {
    "No": 0,
    "Hybrid": 1,
    "Yes": 2,
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        experience_years = int(request.form["experience_years"])
        skills_count = int(request.form["skills_count"])
        certifications = int(request.form["certifications"])

        job_title = request.form["job_title"].strip()
        education_level = request.form["education_level"].strip().title()
        industry = request.form["industry"].strip()
        company_size = request.form["company_size"].strip().title()
        location = request.form["location"].strip()
        remote_work = request.form["remote_work"].strip().title()

        experience_bucket = exp_bucket(experience_years)
        skill_strength = skills_count + (certifications * 2)
        exp_skill_interaction = experience_years * skills_count

        education_level_encoded = education_map.get(education_level)
        company_size_encoded = company_map.get(company_size)
        remote_encoded = remote_map.get(remote_work)

        if education_level_encoded is None:
            raise ValueError(f"Invalid education level: {education_level}")

        if company_size_encoded is None:
            raise ValueError(f"Invalid company size: {company_size}")

        if remote_encoded is None:
            raise ValueError(f"Invalid remote work value: {remote_work}")

        experience_company = experience_years * company_size_encoded
        experience_education = experience_years * education_level_encoded

        input_df = pd.DataFrame([{
            "job_title": job_title,
            "experience_years": experience_years,
            "education_level": education_level,
            "skills_count": skills_count,
            "industry": industry,
            "company_size": company_size,
            "location": location,
            "remote_work": remote_work,
            "certifications": certifications,
            "experience_bucket": experience_bucket,
            "skill_strength": skill_strength,
            "exp_skill_interaction": exp_skill_interaction,
            "education_level_encoded": education_level_encoded,
            "company_size_encoded": company_size_encoded,
            "remote_encoded": remote_encoded,
            "experience_company": experience_company,
            "experience_education": experience_education
        }])

        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Salary: {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)