# simulator.py
import pandas as pd
import requests
import time
import json

# !!! IMPORTANT: Replace this with your n8n Webhook Test URL !!!
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/predict"


def simulate_transactions():
    try:
        df = pd.read_csv("creditcard.csv")
    except FileNotFoundError:
        print("Error: creditcard.csv not found...")
        return

    sample_data = pd.concat([
        df[df['Class'] == 0].head(10),
        df[df['Class'] == 1].head(1)
    ]).sample(frac=1).reset_index(drop=True)

    print("Starting transaction simulation with SUPERVISED model...")
    for index, row in sample_data.iterrows():
        # --- SEND ALL FEATURES ---
        transaction = row.drop('Class').to_dict()
        # --- END CHANGE ---

        is_known_fraud = row['Class'] == 1

        try:
            print(
                f"--> Sending transaction {index + 1} (Known Fraud: {is_known_fraud}) | Amount: ${transaction['Amount']:.2f}")
            response = requests.post(N8N_WEBHOOK_URL, json=transaction)
            response.raise_for_status()
            print(f"<-- Received response: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending transaction: {e}")

        time.sleep(2)


if __name__ == "__main__":
    simulate_transactions()