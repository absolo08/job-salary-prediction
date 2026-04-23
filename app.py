
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('DC.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -------- Numeric inputs --------
        job_title = float(request.form.get("job_title", 0))
        experience_years = float(request.form.get('experience_years', 0))
        skills_count = float(request.form.get('skills_count', 0))
        certifications = float(request.form.get('certifications', 0))

        # -------- Education --------
        edu_diploma = int(request.form.get('edu_diploma', 0))
        edu_highschool = int(request.form.get('edu_highschool', 0))
        edu_master = int(request.form.get('edu_master', 0))
        edu_phd = int(request.form.get('edu_phd', 0))

        # -------- Industry --------
        ind_education = int(request.form.get('ind_education', 0))
        ind_finance = int(request.form.get('ind_finance', 0))
        ind_government = int(request.form.get('ind_government', 0))
        ind_healthcare = int(request.form.get('ind_healthcare', 0))
        ind_manufacturing = int(request.form.get('ind_manufacturing', 0))
        ind_media = int(request.form.get('ind_media', 0))
        ind_retail = int(request.form.get('ind_retail', 0))
        ind_tech = int(request.form.get('ind_tech', 0))
        ind_telecom = int(request.form.get('ind_telecom', 0))

        # -------- Company size --------
        comp_large = int(request.form.get('comp_large', 0))
        comp_medium = int(request.form.get('comp_medium', 0))
        comp_small = int(request.form.get('comp_small', 0))
        comp_startup = int(request.form.get('comp_startup', 0))

        # -------- Location --------
        loc_canada = int(request.form.get('loc_canada', 0))
        loc_germany = int(request.form.get('loc_germany', 0))
        loc_india = int(request.form.get('loc_india', 0))
        loc_netherlands = int(request.form.get('loc_netherlands', 0))
        loc_remote = int(request.form.get('loc_remote', 0))
        loc_singapore = int(request.form.get('loc_singapore', 0))
        loc_sweden = int(request.form.get('loc_sweden', 0))
        loc_uk = int(request.form.get('loc_uk', 0))
        loc_usa = int(request.form.get('loc_usa', 0))

        # -------- Remote --------
        remote_no = int(request.form.get('remote_no', 0))
        remote_yes = int(request.form.get('remote_yes', 0))

        # -------- Feature vector (ORDER MUST MATCH TRAINING) --------
        features = np.array([[ 
            job_title,experience_years, skills_count, certifications,
            edu_diploma, edu_highschool, edu_master, edu_phd,
            ind_education, ind_finance, ind_government, ind_healthcare,
            ind_manufacturing, ind_media, ind_retail, ind_tech, ind_telecom,
            comp_large, comp_medium, comp_small, comp_startup,
            loc_canada, loc_germany, loc_india, loc_netherlands,
            loc_remote, loc_singapore, loc_sweden, loc_uk, loc_usa,
            remote_no, remote_yes
        ]])

        # -------- Prediction --------
        pred = model.predict(features)
        salary = round(pred[0], 2)

        return render_template('index.html', prediction=f"Estimated Salary: ${salary}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)