import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
import uvicorn
from typing import Dict
from pydantic import BaseModel, Field
from enum import Enum
from typing_extensions import Annotated

MODEL_PATH = "./model/catmodel.cbm"

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"

class YesNoEnum(str,Enum):
    yes = "Yes"
    no = "No"

class InternetEnum(str, Enum):
    fiber_o = "Fiber optic"
    dsl = "DSL"
    no = "No"


class ContractEnum(str, Enum):
    one = "Two year"
    two = "One year"
    three = "Month-to-month"

class PaymentEnum(str, Enum):
    one = "Electronic check"
    two = "Mailed check"
    three = "Bank transfer (automatic)"


d = pd.read_csv("./X_train.csv")

max_tenure = d['tenure'].max()
max_monthly = d["MonthlyCharges"].max()
max_total = d["TotalCharges"].max()



class Customer(BaseModel):
    gender: GenderEnum 
    SeniorCitizen: Annotated[int, Field(strict=True, ge=0,le=1)]
    Partner: YesNoEnum
    Dependents: YesNoEnum
    tenure: Annotated[float, Field(strict=True, ge=0,le=max_tenure)]
    PhoneService: YesNoEnum
    MultipleLines: YesNoEnum
    InternetService: InternetEnum
    OnlineSecurity: YesNoEnum
    OnlineBackup: YesNoEnum
    DeviceProtection: YesNoEnum
    TechSupport: YesNoEnum
    StreamingTV: YesNoEnum
    StreamingMovies: YesNoEnum
    Contract: ContractEnum
    PaperlessBilling: YesNoEnum
    PaymentMethod: PaymentEnum
    MonthlyCharges: Annotated[float, Field(strict=True, ge=0,le=max_monthly)]
    TotalCharges: Annotated[float, Field(strict=True, ge=0,le=max_total)]
    first_month_tenure: Annotated[int, Field(strict=True, ge=0,le=1)]
    lowMonthlyCharges: Annotated[int, Field(strict=True, ge=0,le=1)]


def load_model():
    cb = CatBoostClassifier()
    model = cb.load_model(MODEL_PATH)
    return model

def convert_data(data:Dict): 
    df = pd.DataFrame.from_dict([data])
    return df



def predict(data, model):
    df = convert_data(data)

    churn_prob = model.predict_proba(df)[0][1]

    return churn_prob



model = load_model()

app = FastAPI(title="Churn Prediction", version="1.0")

@app.get('/')
def index():
    return {'message': 'CHURN Prediction API'}

@app.post('/create_customer/')
async def create_cust(customer: Customer):
    return customer


@app.post('/predict/')
def predict_churn(data: Dict):
    churn_p = predict(data,model)


    #return churn probabilty
    return {'Churn Probability': churn_p}


if __name__ == '__main__':
    uvicorn.run("fast-api:app", host='127.0.0.1', port=5000)

