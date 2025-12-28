# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Încarcă modelul și encoderele
with open("model/salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/le_country.pkl", "rb") as f:
    le_country = pickle.load(f)

with open("model/le_education.pkl", "rb") as f:
    le_education = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        country = request.form["country"]
        education = request.form["education"]
        experience = float(request.form["experience"])

        # Transformare inputuri
        country_encoded = le_country.transform([country])[0]
        education_encoded = le_education.transform([education])[0]

        X = np.array([[country_encoded, education_encoded, experience]])
        predicted_salary = model.predict(X)[0]
        prediction = f"${predicted_salary:,.0f} / year"

    # Pentru popularea select-urilor în HTML
    countries = sorted(le_country.classes_)
    educations = sorted(le_education.classes_)

    return render_template("index.html", prediction=prediction,
                           countries=countries, educations=educations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
