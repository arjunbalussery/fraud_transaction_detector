import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- 1. SETUP ---
app = FastAPI()

# In-memory storage for our trained model and scaler
# In a real app, you would load this from a file (e.g., using joblib or pickle)
model_storage = {}

# Pydantic model for input validation (like a DTO)
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# --- 2. MODEL TRAINING ENDPOINT ---
@app.post("/train")
def train_model():
    """
    Loads data, trains a SUPERVISED Logistic Regression model.
    """
    global model_storage
    df = pd.read_csv("creditcard.csv")

    # Define features (X) and the target (y)
    features = [col for col in df.columns if col not in ['Class']]
    X = df[features]
    y = df['Class']  # This is the target label we want to predict

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- MODEL CHANGE ---
    # Use a supervised model. class_weight='balanced' helps with the imbalanced data.
    model = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear')
    model.fit(X_scaled, y)  # Train the model to predict y from X
    # --- END MODEL CHANGE ---

    model_storage['model'] = model
    model_storage['scaler'] = scaler
    model_storage['features'] = features

    return {"message": f"SUPERVISED model trained successfully on {len(features)} features."}


# --- 3. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """
    Predicts fraud using the new supervised model.
    """
    if 'model' not in model_storage:
        return {"error": "Model not trained yet. Please call /train first."}

    model = model_storage['model']
    scaler = model_storage['scaler']
    feature_order = model_storage['features']

    transaction_data = pd.DataFrame([transaction.dict()])
    transaction_data = transaction_data[feature_order]

    transaction_scaled = scaler.transform(transaction_data)

    # --- PREDICTION LOGIC CHANGE ---
    # .predict() now returns 0 (legit) or 1 (fraud)
    prediction = model.predict(transaction_scaled)

    # .predict_proba() gives the probability for each class [prob_legit, prob_fraud]
    probability = model.predict_proba(transaction_scaled)

    is_fraud = prediction[0] == 1
    fraud_probability = probability[0][1]  # Get the probability of it being fraud
    # --- END PREDICTION LOGIC CHANGE ---

    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(fraud_probability)
    }