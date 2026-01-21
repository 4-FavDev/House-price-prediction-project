from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            OverallQual = float(request.form["OverallQual"])
            GrLivArea = float(request.form["GrLivArea"])
            TotalBsmtSF = float(request.form["TotalBsmtSF"])
            GarageCars = float(request.form["GarageCars"])
            BedroomAbvGr = float(request.form["BedroomAbvGr"])
            YearBuilt = float(request.form["YearBuilt"])

            features = np.array([[ 
                OverallQual,
                GrLivArea,
                TotalBsmtSF,
                GarageCars,
                BedroomAbvGr,
                YearBuilt
            ]])

            prediction = model.predict(features)[0]
            prediction = f"₦{prediction:,.2f}"

        except:
            prediction = "Invalid input. Please try again."

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
