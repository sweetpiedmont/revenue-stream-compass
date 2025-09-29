import os
import requests
import json

AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = "appgHTZhO9XkeAESy"   # replace with your base ID
TABLE_ID = "tblsH62BjRZiXUaWV" # replace with your table ID

def fetch_user_scores(user_id):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": f"{{User ID}} = '{user_id}'", "maxRecords": 1}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    records = resp.json()["records"]
    if not records:
        raise ValueError(f"No record found for User ID {user_id}")
    record = records[0]["fields"]

    # Pull JSON text and parse
    json_text = record.get("User Scores JSON", "{}")
    scores = json.loads(json_text)

    return scores

if __name__ == "__main__":
    user_id = "38883ef1-473d-4bcd-a89e-b6e5ef472c01"
    scores = fetch_user_scores(user_id)
    print("✅ Parsed user scores:", scores)
    print("Customer service score is:", scores.get("customer_service"))
