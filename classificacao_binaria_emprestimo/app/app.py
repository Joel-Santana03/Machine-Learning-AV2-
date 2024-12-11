from flask import Flask, render_template, request
import joblib
import numpy as np

# Inicializar a aplicação Flask
app = Flask(__name__)

# Carregar o modelo salvo
model_path = "final_logistic_regression_pipeline.pkl"
model = joblib.load(model_path)

@app.route("/")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Coletar dados do formulário
    gender = 1 if request.form["gender"] == "Male" else 0
    married = 1 if request.form["married"] == "Yes" else 0
    dependents = int(request.form["dependents"])
    education = 1 if request.form["education"] == "Graduate" else 0
    self_employed = 1 if request.form["self_employed"] == "Yes" else 0
    applicant_income = float(request.form["applicant_income"])
    loan_amount = float(request.form["loan_amount"])

    # Criar array de entrada
    input_features = np.array([[gender, married, dependents, education, self_employed, applicant_income, loan_amount]])

    # Logar os dados de entrada
    print("Input Features:", input_features)

    # Fazer predição
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0] if hasattr(model, "predict_proba") else None

    # Logar o resultado da predição
    print("Prediction:", prediction)
    print("Probability:", prediction_proba)

    # Resultado
    result = "Approved" if prediction == 1 else "Rejected"
    prob = f"{max(prediction_proba) * 100:.2f}%" if prediction_proba is not None else "N/A"

    return render_template("result.html", result=result, prob=prob)


if __name__ == "__main__":
    app.run(debug=True)
