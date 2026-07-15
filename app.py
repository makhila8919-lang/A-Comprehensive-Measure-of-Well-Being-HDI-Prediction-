from flask import Flask, render_template, request
import joblib
import numpy as np

from config import SECRET_KEY

from config import MODEL_PATH, ENCODER_PATH


app = Flask(__name__)

app.config["SECRET_KEY"] = SECRET_KEY
# -------------------------------------
# Load Model
# -------------------------------------

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# -------------------------------------
# Recommendation Function
# -------------------------------------

def get_recommendation(category):

    recommendations = {

        "Very High": [
            "Excellent healthcare system.",
            "Maintain education quality.",
            "Continue sustainable economic growth.",
            "Invest in innovation."
        ],

        "High": [
            "Improve higher education.",
            "Strengthen healthcare infrastructure.",
            "Increase employment opportunities.",
            "Boost research and development."
        ],

        "Medium": [
            "Increase literacy rate.",
            "Improve healthcare facilities.",
            "Reduce poverty.",
            "Increase public investment."
        ],

        "Low": [
            "Increase school enrollment.",
            "Improve access to hospitals.",
            "Reduce infant mortality.",
            "Create employment opportunities."
        ]

    }

    return recommendations.get(
    category,
    ["No recommendation available."])


# -------------------------------------
# Home Page
# -------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------------------
# Prediction
# -------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        life = float(request.form["life"])

        expected = float(request.form["expected"])

        mean = float(request.form["mean"])

        gni = float(request.form["gni"])

        features = np.array([[

            life,
            expected,
            mean,
            gni

        ]])

        prediction = model.predict(features)

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(features)

            confidence = round(np.max(probability) * 100, 2)

        else:

            confidence = 100

        category = encoder.inverse_transform(prediction)[0]

        recommendation = get_recommendation(category)

        return render_template(

            "result.html",

            prediction=category,

            confidence=confidence,

            recommendation=recommendation,

            life=life,

            expected=expected,

            mean=mean,

            gni=gni

        )

    except Exception as e:

        return render_template(

            "result.html",

            prediction="Error",

            confidence=0,

            recommendation=[str(e)]

        )


# -------------------------------------
# About Page
# -------------------------------------

@app.route("/about")
def about():

    return render_template("about.html")


# -------------------------------------
# Contact Page
# -------------------------------------

@app.route("/contact")
def contact():

    return render_template("contact.html")


# -------------------------------------
# Main
# -------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
            
    )
