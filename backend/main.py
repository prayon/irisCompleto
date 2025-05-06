from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Cargar modelo
model = joblib.load("iris_model.joblib")
app = FastAPI()

# Conexión a la base de datos PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS iris_predictions (
    id SERIAL PRIMARY KEY,
    sepal_length FLOAT,
    sepal_width FLOAT,
    petal_length FLOAT,
    petal_width FLOAT,
    prediction INT
);
""")
conn.commit()

# Esquema de entrada
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    values = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(values)[0])

    # Guardar en base de datos
    cursor.execute("""
    INSERT INTO iris_predictions (sepal_length, sepal_width, petal_length, petal_width, prediction)
    VALUES (%s, %s, %s, %s, %s);
    """, (data.sepal_length, data.sepal_width, data.petal_length, data.petal_width, prediction))
    conn.commit()

    return {"prediction": prediction}
