from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from CONFIGS.paths import *
from SRC.analysis import *
from SRC.preprocessing import load_dataset
from SRC.splitting import split_dataset
from SRC.cleaning import preprocess_dataset
df = load_dataset()
df = preprocess_dataset(df)
df = df.dropna(subset=["loan_status"])
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)



app = FastAPI()

model = joblib.load(RISK_MODEL)

class LoanData(BaseModel):
    loan_amnt: float
    term: float
    int_rate: float
    installment: float
    grade: str
    emp_length: float
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    delinq_2yrs: float
    fico_range_low: float
    fico_range_high: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float



@app.post("/predict")
def predict(data: LoanData):
    input_df = pd.DataFrame([data.model_dump()])
    input_df = pd.get_dummies(input_df)
    expected_columns = model.feature_names_in_
    input_df = input_df.reindex(
        columns=expected_columns,
        fill_value=0
    )
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_index = list(model.classes_).index(1)
    approval_probability = probabilities[class_index] * 100
    explanation = explain_prediction(
        input_df,
        model,
        X_train)
    return {
        "prediction": int(prediction),
        "approval_probability": round(
            float(approval_probability), 2),
        "feature_influence": explanation.to_dict(
            orient="records")
    }
