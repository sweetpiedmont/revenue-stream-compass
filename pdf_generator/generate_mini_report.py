# generate_mini_report.py

import os
import sys
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from save_to_drive import save_navigation_planner  # if you still need this helper

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
        return [x]
    else:
        return []

def fetch_user_by_id(user_id):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {
        "filterByFormula": f"{{User ID}} = '{user_id}'",
        "maxRecords": 1
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    records = resp.json()["records"]
    if not records:
        raise ValueError(f"No record found for User ID {user_id}")
    record = records[0]["fields"]

    # Force Top5 fields into lists
    top5_names = ensure_list(record.get("Top5 Name", []))
    top5_narratives = ensure_list(record.get("Top5 Short Narrative", []))
    top5 = [{"name": n, "narrative": t} for n, t in zip(top5_names, top5_narratives)]

    # Force All Streams fields into lists
    all_names = ensure_list(record.get("All Streams Name", []))
    all_ranks = ensure_list(record.get("All Streams Rank", []))
    all_streams = sorted(
        [{"name": n, "rank": r} for n, r in zip(all_names, all_ranks)],
        key=lambda x: x["rank"]
    )

    return {
        "user_id": record.get("User ID", "test123"),
        "user_name": record.get("First Name", "Test User"),
        "top5": top5,
        "all_streams": all_streams,
    }

def render_mini_report_html(user_name, top5, all_streams):
    """Render Mini-Report HTML from Jinja template with given data."""
    template = env.get_template("mini_report.html")
    return template.render(user_name=user_name, top5=top5, all_streams=all_streams)

def generate_mini_report(user_id, user_name, top5, all_streams, outpath="mini_report.pdf"):
    """Generate Mini-Report PDF and save locally + to Google Drive."""
    html_content = render_mini_report_html(user_name, top5, all_streams)

    # Local write
    HTML(string=html_content).write_pdf(outpath)
    print(f"✅ Mini-Report written to {outpath}")

    # Google Drive write
    drive_folder = "/Users/sharon/Library/CloudStorage/GoogleDrive-hello@sweetpiedmontacademy.com/My Drive/Mini Report Storage"
    folder = Path(drive_folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"mini_report_{user_id}.pdf"
    outpath_drive = folder / filename
    HTML(string=html_content).write_pdf(str(outpath_drive))
    print(f"✅ Saved Mini-Report for user {user_id} at {outpath_drive}")

if __name__ == "__main__":
    user_data = fetch_latest_user()
    generate_mini_report(
        user_data["user_id"],
        user_data["user_name"],
        user_data["top5"],
        user_data["all_streams"],
    )
