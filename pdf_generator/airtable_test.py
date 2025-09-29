import os
import requests

# Grab your env var
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")

# Replace with your actual IDs from the URL
BASE_ID = "appgHTZhO9XkeAESy"
TABLE_ID = "tblsH62BjRZiXUaWV"

url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"

headers = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}"
}

# Fetch just 1 row to test
params = {
    "maxRecords": 1
}

resp = requests.get(url, headers=headers, params=params)
print("Status code:", resp.status_code)

if resp.ok:
    data = resp.json()
    print("✅ Connection works!")
    print("First record fields:")
    print(data["records"][0]["fields"])
else:
    print("❌ Error:", resp.text)
