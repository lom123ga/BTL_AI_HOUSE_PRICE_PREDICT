from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from pydantic import BaseModel
from model import load_model
from sklearn.preprocessing import PolynomialFeatures

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load model weights
w = load_model()

class FeatureData(BaseModel):
    sqft_living: float
    grade: float
    sqft_living15: float
    bathrooms: float
    view: float
    sqft_basement: float
    yr_renovated: float
    lat: float
    waterfront: float
    yr_built: float
    bedrooms: float
    year: float
    month: float
    day: float
    floors: float

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, 
            sqft_living: float = Form(...),
            grade: float = Form(...),
            sqft_living15: float = Form(...),
            bathrooms: float = Form(...),
            view: float = Form(...),
            sqft_basement: float = Form(...),
            yr_renovated: float = Form(...),
            lat: float = Form(...),
            waterfront: float = Form(...),
            yr_built: float = Form(...),
            bedrooms: float = Form(...),
            year: float = Form(...),
            month: float = Form(...),
            day: float = Form(...),
            floors: float = Form(...)):

    features = np.array([1, sqft_living, grade, sqft_living15, bathrooms, view, sqft_basement,
                         yr_renovated, lat, waterfront, yr_built, bedrooms, year, month, day, floors])
    features = PolynomialFeatures(degree=3).fit_transform([features])
    prediction = np.dot(features, w)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction[0]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)