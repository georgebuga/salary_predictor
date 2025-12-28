# app.py
import flask
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model
with open("model/salary_model.pkl", "rb") as f:
    model = pickle.load(f)

# Helper: encode input like in training
def preprocess_input(country, education, experience):
    # Categorial mappings (example, update based on your model!)
    country_map = {
        "United States": 0,
        "Germany": 1,
        "India": 2,
        # Add other countries...
    }
    education_map = {
        "Less than a Bachelors": 0,
        "Bachelor’s degree": 1,
        "Master’s degree": 2,
        "Post grad": 3,
    }

    country_encoded = country_map.get(country, -1)
    education_encoded = education_map.get(education, -1)
    return np.array([[country_encoded, education_encoded, experience]])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        country = request.form["country"]
        education = request.form["education"]
        experience = float(request.form.get("yearsExperience", 0))

        input_data = preprocess_input(country, education, experience)
        salary = model.predict(input_data)[0]
        prediction = f"{salary:,.0f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
