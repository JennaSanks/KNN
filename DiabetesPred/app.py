from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained Diabetes KNN pipeline
model = load("KNNModel.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get the 8 Diabetes features from the HTML form
    pregnancies = float(request.form["Pregnancies"])
    glucose = float(request.form["Glucose"])
    blood_pressure = float(request.form["BloodPressure"])
    skin_thickness = float(request.form["SkinThickness"])
    insulin = float(request.form["Insulin"])
    bmi = float(request.form["BMI"])
    diabetes_pedigree = float(
        request.form["DiabetesPedigreeFunction"]
    )
    age = float(request.form["Age"])

    # Create a DataFrame
    # IMPORTANT:
    # Column names and order must match the training data
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability of Diabetes
    probability = model.predict_proba(input_data)[0][1]

    # Convert numerical prediction to readable result
    if prediction == 1:
        result = "Diabetes"
    else:
        result = "No Diabetes"

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}",
        probability=f"{probability:.2%}"
    )


if __name__ == "__main__":
    app.run(debug=True)
