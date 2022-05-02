
import json
from plistlib import load
from typing import Optional
from typing_extensions import Self
from DataModel import DataModel, DataModelList, DataModelPredict
from typing import FrozenSet
import pandas as pd
import numpy as np
from pandas import json_normalize
from sklearn.metrics import r2_score, mean_squared_error
from joblib import load
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def make_predictions(data: DataModelList):
    an = jsonable_encoder(data)
    df = json_normalize(an['data'])
    df.columns = DataModel.columns()
    model = load("assets/pipelinef.joblib")
    result = model.predict(df)
    lists = result.tolist()
    json_predict = json.dumps(lists)
    return {"predict": json_predict}

@app.post("/r2")
def r2(data: DataModelList, dataTrue: DataModelPredict):
    an = jsonable_encoder(data)
    df = json_normalize(an['data'])
    df.columns = DataModel.columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    dict = jsonable_encoder(dataTrue)
    y_true = []
    for i in dict["dataTrue"]:
        y_true.append(float(i["life_expectancy"]))
    r2 = r2_score(y_true, result.tolist())
    return {"r2": r2}
