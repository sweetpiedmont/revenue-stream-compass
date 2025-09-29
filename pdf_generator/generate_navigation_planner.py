# generate_navigation_planner.py

import os
import sys
import json
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd
import numpy as np
import base64

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rsc_from_excel import (
    load_from_excel,
    get_channel_long_narrative,
    render_navigation_page,
    save_navigation_planner_pdf,
    slugify
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
    try:
        decoded = base64.b64decode(raw_scores).decode("utf-8")
        user_scores = json.loads(decoded)
    except Exception as e:
        print(f"⚠️ Failed to decode scores, using empty: {e}")
        user_scores = {}
    
    return record, user_scores

# DEBUG Code
print("User scores from Airtable:", list(user_scores.keys())[:5])
print("Example factor slugs from Excel:", [slugify(f) for f in factors["factor_name"].head(5)])

def calculate_rankings(user_scores, channels):

    factor_cols = [c for c in channels.columns if c.startswith("f_")]

    # Align user_scores with channel factor columns
    uw = {f"f_{fid}": float(user_scores.get(fid, 0)) for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])
    uw_aligned = uw_df.reindex(columns=channels[factor_cols].columns, fill_value=0)

    ch = channels.copy()

    # --- Fit score (weighted avg) ---
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

        weighted_avg = np.dot(scores, weights) / weights.sum()
        score = weighted_avg / 10.0  # normalize
        adjusted_scores.append(score)

    ch["fit_score"] = adjusted_scores

    # --- Coverage score ---
    max_factors_high = (channels[factor_cols] >= 8).sum(axis=1).max()
    coverages = []
    for idx, row in channels.iterrows():
        k = (row[factor_cols] >= 8).sum()
        coverage = k / max_factors_high if max_factors_high > 0 else 0
        coverages.append(coverage)
    ch["coverage"] = coverages

    # --- Blend ---
    ch["blend_score_70"] = 0.7 * ch["fit_score"] + 0.3 * ch["coverage"]
    ch["score"] = ch["blend_score_70"]

    # --- Ranked output ---
    rackstack = (
        ch.loc[:, ["channel_name", "fit_score", "coverage", "blend_score_70", "score"]]
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

    return rackstack

def generate_navigation_planner(user_id, user_name, user_scores, outpath="navigation_planner.pdf"):
    """Generate Navigation Planner PDF for this user."""
    # Load data from Excel
    factors, categories, channels, narratives = load_from_excel(
        Path("Extreme_Weighting_Scoring_Prototype_for_FormWise_REPAIRED.xlsx")
    )

    # Build factor → category color map
    factor_to_category = dict(zip(factors["factor_name"], factors["category_name"]))
    category_to_color = dict(zip(categories["category_name"], categories["category_color"]))
    factor_to_color = {
        f: category_to_color.get(cat, "#cccccc")
        for f, cat in factor_to_category.items()
    }

    # --- Build all 18 pages in RANKED ORDER ---
    rackstack = calculate_rankings(user_scores, channels)
    
    planner_pages = []
    for i, row in rackstack.iterrows():
        ch_name = row["channel_name"]
        rank = i + 1

        # Long Narrative (2 short paragraphs)
        narrative = get_channel_long_narrative(ch_name, narratives, user_scores)

        # Advantages/obstacles (weight >= 4 factors)
        df = narratives[(narratives["channel_name"] == ch_name) & (narratives["weight"] >= 4)].copy()
        df["factor_id"] = df["factor_name"].map(slugify)
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

    # Save PDF once
    filename = f"navigation_planner_{user_id}.pdf"
    local_path = Path(filename)
    save_navigation_planner_pdf(planner_pages, str(local_path))

    # Copy to Drive
    drive_folder = Path("/Users/sharon/Library/CloudStorage/GoogleDrive-hello@sweetpiedmontacademy.com/My Drive/Navigation Planner Storage")
    drive_folder.mkdir(parents=True, exist_ok=True)
    drive_path = drive_folder / filename
    local_path.replace(drive_path)  # move file instead of regenerating

    print(f"✅ Navigation Planner written to {drive_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = "test123"

    record, user_scores = fetch_user_scores(user_id)
    user_name = record.get("First Name", "Test User")

    generate_navigation_planner(user_id, user_name, user_scores)