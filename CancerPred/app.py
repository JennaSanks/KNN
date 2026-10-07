from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load trained KNN model
model = load("KNNModel.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -------------------------------------------------
    # Get the 30 input features from the HTML form
    # -------------------------------------------------

    input_data = pd.DataFrame([{

        "radius_mean": float(request.form["radius_mean"]),
        "texture_mean": float(request.form["texture_mean"]),
        "perimeter_mean": float(request.form["perimeter_mean"]),
        "area_mean": float(request.form["area_mean"]),
        "smoothness_mean": float(request.form["smoothness_mean"]),
        "compactness_mean": float(request.form["compactness_mean"]),
        "concavity_mean": float(request.form["concavity_mean"]),
        "concave points_mean": float(request.form["concave_points_mean"]),
        "symmetry_mean": float(request.form["symmetry_mean"]),
        "fractal_dimension_mean": float(
            request.form["fractal_dimension_mean"]
        ),

        "radius_se": float(request.form["radius_se"]),
        "texture_se": float(request.form["texture_se"]),
        "perimeter_se": float(request.form["perimeter_se"]),
        "area_se": float(request.form["area_se"]),
        "smoothness_se": float(request.form["smoothness_se"]),
        "compactness_se": float(request.form["compactness_se"]),
        "concavity_se": float(request.form["concavity_se"]),
        "concave points_se": float(
            request.form["concave_points_se"]
        ),
        "symmetry_se": float(request.form["symmetry_se"]),
        "fractal_dimension_se": float(
            request.form["fractal_dimension_se"]
        ),

        "radius_worst": float(request.form["radius_worst"]),
        "texture_worst": float(request.form["texture_worst"]),
        "perimeter_worst": float(request.form["perimeter_worst"]),
        "area_worst": float(request.form["area_worst"]),
        "smoothness_worst": float(request.form["smoothness_worst"]),
        "compactness_worst": float(request.form["compactness_worst"]),
        "concavity_worst": float(request.form["concavity_worst"]),
        "concave points_worst": float(
            request.form["concave_points_worst"]
        ),
        "symmetry_worst": float(request.form["symmetry_worst"]),
        "fractal_dimension_worst": float(
            request.form["fractal_dimension_worst"]
        )
    }])

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    prediction = model.predict(input_data)[0]

    # Probability, if the trained model supports it
    try:
        probability = model.predict_proba(input_data)[0]

        # Probability of malignant class
        if hasattr(model, "classes_"):
            classes = model.classes_
        else:
            classes = None

        if classes is not None and "M" in classes:
            malignant_probability = probability[
                list(classes).index("M")
            ]
        elif classes is not None and 1 in classes:
            malignant_probability = probability[
                list(classes).index(1)
            ]
        else:
            malignant_probability = None

    except AttributeError:
        malignant_probability = None

    # -------------------------------------------------
    # Convert prediction into readable result
    # -------------------------------------------------

    if prediction == "M" or prediction == 1:
        result = "Malignant"
        result_class = "danger"
    else:
        result = "Benign"
        result_class = "safe"

    # -------------------------------------------------
    # Send result back to HTML
    # -------------------------------------------------

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}",
        result_class=result_class,
        probability=(
            f"{malignant_probability:.2%}"
            if malignant_probability is not None
            else None
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
