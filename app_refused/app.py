from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd


class LoanApplication(BaseModel):
    SK_ID_PREV: int
    AMT_ANNUITY: float
    AMT_APPLICATION: float
    AMT_CREDIT: float
    AMT_DOWN_PAYMENT: float
    AMT_GOODS_PRICE: float
    WEEKDAY_APPR_PROCESS_START: str
    HOUR_APPR_PROCESS_START: int
    FLAG_LAST_APPL_PER_CONTRACT: str
    NFLAG_LAST_APPL_IN_DAY: int
    RATE_DOWN_PAYMENT: float
    NAME_CASH_LOAN_PURPOSE: str
    DAYS_DECISION: int
    NAME_PAYMENT_TYPE: str
    NAME_TYPE_SUITE: str
    NAME_CLIENT_TYPE: str
    NAME_GOODS_CATEGORY: str
    NAME_PORTFOLIO: str
    NAME_PRODUCT_TYPE: str
    CHANNEL_TYPE: str
    SELLERPLACE_AREA: int
    NAME_SELLER_INDUSTRY: str
    CNT_PAYMENT: float
    NAME_YIELD_GROUP: str
    PRODUCT_COMBINATION: str


class Refused(BaseModel):
    refused: int
    probability: float


preprocessor = joblib.load("../models/refused_preprocessor_V1.joblib")
model = joblib.load("../models/refused_model_V1.joblib")
PROFITABLE_THRESHOLD = 0.35


app = FastAPI()


@app.post("/predict_refused", response_model=Refused)
def predict_refused(payload: LoanApplication):
    X = pd.DataFrame([payload.model_dump()])
    X_tf = preprocessor.transform(X)
    pred_proba = model.predict_proba(X_tf)[0][1]
    pred = 1 if pred_proba > PROFITABLE_THRESHOLD else 0
    result = dict(refused=pred, probability=pred_proba)
    return result
