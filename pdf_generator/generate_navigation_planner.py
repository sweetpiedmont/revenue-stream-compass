# generate_navigation_planner.py

import os
import sys
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# Airtable config
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = "appgHTZhO9XkeAESy"
TABLE_ID = "tblsH62BjRZiXUaWV"

# --- Jinja2 setup ---
env = Environment(loader=FileSystemLoader("pdf_generator/templates"))

def ensure_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [item.strip() for item in x.split(",") if item.strip()]
    else:
        return []

def fetch_user_by_id(user_id):
    """Fetch user record from Airtable by User ID."""
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": f"{{User ID}} = '{user_id}'", "maxRecords": 1}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    records = resp.json()["records"]
    if not records:
        raise ValueError(f"No record found for User ID {user_id}")
    record = records[0]["fields"]

    return {
        "user_id": record.get("User ID", "test123"),
        "user_name": record.get("First Name", "Test User"),
        # Navigation Planner will probably need more fields than Top5/All Streams,
        # but let's pull these for now as placeholders:
        "top5": list(zip(
            ensure_list(record.get("Top5 Name", [])),
            ensure_list(record.get("Top5 Short Narrative", []))
        )),
        "all_streams": sorted(
            [{"name": n, "rank": r} for n, r in zip(
                ensure_list(record.get("All Streams Name", [])),
                ensure_list(record.get("All Streams Rank", []))
            )],
            key=lambda x: x["rank"]
        )
    }

def render_navigation_html(user_name, top5, channels):
    """Render HTML from Jinja template with given data."""
    template = env.get_template("planner.html")
    return template.render(
        user_name=user_name,
        top5=top5,
        channels=channels,
    )

def generate_pdf(user_id, user_name, top5, channels, outpath="planner.pdf"):
    """
    Generate the Navigation Planner PDF and save both locally and to Drive.
    """
    html_content = render_navigation_html(user_name, top5, channels)

    # Write locally
    HTML(string=html_content).write_pdf(outpath)
    print(f"✅ PDF written to {outpath}")

    # ALSO write to Drive
    drive_folder = "/Users/sharon/Library/CloudStorage/GoogleDrive-hello@sweetpiedmontacademy.com/My Drive/Navigation Planner Storage"
    save_navigation_planner(user_id, html_content, drive_folder)


if __name__ == "__main__":
    # Allow user_id to be passed from command line
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = "test123"  # fallback default

    # Fake test data (until hooked into Airtable)
    user_name = "Test User"
    top5 = ["Workshops", "Full Service Weddings", "Farmers Markets", "Subscriptions", "DIY Buckets"]

    channels = [
        {
            "name": "Workshops",
            "score": 0.95,
            "narrative": "Workshops are a great way to engage directly with your community and generate steady income.",
            "advantages": ["Direct customer interaction", "High margins", "Community building"],
            "obstacles": ["Requires teaching skills", "Marketing is critical", "Seasonal interest may vary"],
        },
        {
            "name": "Full Service Weddings",
            "score": 0.92,
            "narrative": "Full service weddings provide high revenue but demand detailed planning and reliable staffing.",
            "advantages": ["High profit potential", "Strong demand", "Creative freedom"],
            "obstacles": ["High stress events", "Lots of moving parts", "Requires design expertise"],
        },
    ]

    generate_pdf(user_id, user_name, top5, channels)
