# Fraud Transaction Detector with n8n Workflow

This project demonstrates a machine learning pipeline for real-time fraud detection, orchestrated through an n8n workflow. It was created to showcase the integration of a Python-based machine learning model with n8n for complex workflow automation.

When a transaction is flagged as fraudulent, the n8n workflow calls the Gemini API to provide additional context or information about the nature of the fraud, enriching the final output.

## Project Workflow

The entire process is designed as an automated workflow:

1.  **Transaction Simulation**: The `stimulator.py` script sends a sample transaction to a specified n8n webhook URL.
2.  **n8n Orchestration**: The n8n workflow receives the transaction and triggers the process.
3.  **Prediction Request**: n8n sends the transaction data to the `POST /predict` endpoint of the FastAPI application.
4.  **Fraud Analysis**: The API uses a pre-trained model to determine if the transaction is fraudulent.
5.  **Conditional Logic**:
    *   **If Not Fraud**: The n8n workflow immediately returns the prediction result.
    *   **If Fraud**: The workflow makes a call to the Gemini API, sending details about the transaction to get descriptive insights into the potential fraud.
6.  **Final Response**: The workflow returns the final result, either the simple prediction or the enriched data from Gemini.

## Core Components

*   `main.py`: A FastAPI application that serves the machine learning model via REST endpoints.
*   `stimulator.py`: A script to simulate real-time transactions and trigger the n8n workflow.
*   `creditcard.csv`: The dataset used for training the fraud detection model.

## Getting Started

### Prerequisites

*   Python 3.7+
*   An active n8n instance (cloud or self-hosted)
*   Access to the Gemini API

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd fraud_transaction_detector
    ```

2.  **Install the required packages:**
    ```bash
    pip install "fastapi[all]" scikit-learn pandas requests
    ```

## Running the Application & Workflow

### 1. Start the FastAPI Server

Run the API server from your terminal. This will host the `/train` and `/predict` endpoints.
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 2. Train the Model

Before making predictions, you must train the model by sending a `POST` request to the `/train` endpoint. This only needs to be done once after starting the server.
```bash
curl -X POST http://127.0.0.1:8000/train
```

### 3. Configure and Run the Simulator

The `stimulator.py` script kicks off the workflow. You must configure it to point to your n8n webhook.

*   **Update `stimulator.py`** with your n8n Test Webhook URL:
    ```python
    # in stimulator.py
    N8N_WEBHOOK_URL = "YOUR_N8N_WEBHOOK_TEST_URL"
    ```

*   **Run the script** from a new terminal:
    ```bash
    python stimulator.py
    ```

## Model Training Process

The `POST /train` endpoint performs the following steps to train the model:

1.  **Loads Data**: It reads the `creditcard.csv` dataset using pandas.
2.  **Feature Scaling**: It uses `StandardScaler` from scikit-learn to normalize the feature set. This ensures that all features contribute equally to the model's performance.
3.  **Model Selection**: A **Logistic Regression** model (`sklearn.linear_model.LogisticRegression`) is used for classification.
4.  **Handling Imbalance**: The dataset is highly imbalanced (many more non-fraudulent transactions than fraudulent ones). The `class_weight='balanced'` parameter is used during model initialization to automatically adjust weights inversely proportional to class frequencies.
5.  **In-Memory Storage**: The trained model and the scaler object are stored in a global dictionary in memory, making them readily available for the `/predict` endpoint.

## API Endpoints

### `POST /train`

*   **Description:** Trains the supervised Logistic Regression model as described above.
*   **Request Body:** None.
*   **Response:**
    ```json
    {
      "message": "SUPERVISED model trained successfully on 30 features."
    }
    ```

### `POST /predict`

*   **Description:** This endpoint is intended to be called by the n8n workflow. It receives a transaction and returns a fraud prediction.
*   **Request Body:** A JSON object representing a single transaction.
*   **Response:** A JSON object containing the `is_fraud` status and the `fraud_probability`.
    ```json
    {
      "is_fraud": true,
      "fraud_probability": 0.987
    }
    ```
