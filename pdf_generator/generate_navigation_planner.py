# generate_navigation_planner.py

import os
import sys
import json
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rsc_from_excel import (
    load_from_excel,
    get_channel_long_narrative,
    render_navigation_page,
    save_navigation_planner_pdf
)

# Airtable config
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = "appgHTZhO9XkeAESy"
TABLE_ID = "tblsH62BjRZiXUaWV"

# --- Jinja2 setup ---
env = Environment(loader=FileSystemLoader("pdf_generator/templates"))

def fetch_user_scores(user_id):
    """Fetch a single Airtable record by user_id and parse user_scores JSON."""
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": f"{{User ID}} = '{user_id}'"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    records = resp.json().get("records", [])
    if not records:
        raise ValueError(f"No Airtable record found for user_id={user_id}")
    record = records[0]["fields"]

    raw_scores = record.get("User Scores JSON", "{}")
    user_scores = json.loads(raw_scores)
    return record, user_scores

def generate_navigation_planner(user_id, user_name, user_scores, outpath="navigation_planner.pdf"):
    """Generate Navigation Planner PDF for this user."""
    # Load data from Excel
    factors, categories, channels, narratives = load_from_excel(
        Path("Extreme_Weighting_Scoring_Prototype_for_FormWise_REPAIRED.xlsx")
    )
    # 🔎 Debug: what columns are in channels?
    print("Channels columns:", channels.columns.tolist()[:30])  # show first 30
    print("Total columns:", len(channels.columns))

    # Build factor → category color map
    factor_to_category = dict(zip(factors["factor_name"], factors["category_name"]))
    category_to_color = dict(zip(categories["category_name"], categories["category_color"]))
    factor_to_color = {
        f: category_to_color.get(cat, "#cccccc")
        for f, cat in factor_to_category.items()
    }

    # --- SCORING & RANKING (same as Streamlit) ---
    factor_cols = [c for c in channels.columns if c.startswith("f_")]
    uw = {f"f_{fid}": float(user_scores[fid]) for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])
    uw_aligned = uw_df[channels[factor_cols].columns]

    adjusted_scores = []
    for idx, row in channels.iterrows():
        row_factors = row[factor_cols]
        factor_mask = row_factors > 0
        if factor_mask.sum() == 0:
            adjusted_scores.append(0)
            continue
        scores = uw_aligned.loc[:, row_factors.index[factor_mask]].values.flatten()
        weights = row_factors[factor_mask].values
        if len(scores) == 0 or weights.sum() == 0:
            adjusted_scores.append(0)
            continue
        weighted_avg = (scores @ weights) / weights.sum()
        score = weighted_avg / 10.0
        adjusted_scores.append(score)

    ch = channels.copy()
    ch["score"] = adjusted_scores

    rackstack = (
        ch.loc[:, ["channel_name", "score"]]
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

    # --- Build all 18 pages in RANKED ORDER ---
    planner_pages = []
    for i, row in rackstack.iterrows():
        ch_name = row["channel_name"]
        rank = i + 1

        # Narrative (2 short paragraphs)
        narrative = get_channel_long_narrative(ch_name, narratives, user_scores)

        # Advantages/obstacles (weight >= 4 factors)
        df = narratives[(narratives["channel_name"] == ch_name) & (narratives["weight"] >= 4)].copy()
        df["factor_id"] = df["factor_name"].str.lower().str.replace(" ", "_")
        df["user_score"] = df["factor_id"].map(lambda fid: user_scores.get(fid, 0))

        advantages = [
            (fname, factor_to_color.get(fname, "#d9fdd3"))
            for fname in df[df["user_score"] >= 7]["factor_name"].tolist()
        ]
        obstacles = [
            (fname, factor_to_color.get(fname, "#fdd3d3"))
            for fname in df[df["user_score"] <= 3]["factor_name"].tolist()
        ]

        page_html = render_navigation_page(
            channel_name=ch_name,
            narrative=narrative,
            advantages=advantages,
            obstacles=obstacles,
            rank=rank,
        )
        planner_pages.append(page_html)

    # Save PDF locally
    filename = f"navigation_planner_{user_id}.pdf"
    save_navigation_planner_pdf(planner_pages, filename)

    # Save copy to Google Drive
    drive_folder = "/Users/sharon/Library/CloudStorage/GoogleDrive-hello@sweetpiedmontacademy.com/My Drive/Navigation Planner Storage"
    Path(drive_folder).mkdir(parents=True, exist_ok=True)
    outpath_drive = Path(drive_folder) / filename
    save_navigation_planner_pdf(planner_pages, str(outpath_drive))

    print(f"✅ Navigation Planner written to {filename}")
    print(f"✅ Saved Navigation Planner for user {user_id} at {outpath_drive}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = "test123"

    record, user_scores = fetch_user_scores(user_id)
    user_name = record.get("First Name", "Test User")

    generate_navigation_planner(user_id, user_name, user_scores)