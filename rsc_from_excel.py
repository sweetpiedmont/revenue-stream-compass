import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests #needed for posting to Zapier
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def strength_narrative(score, factor, base_blurb):
    #if score >= 7:
        #return f"Your strong {factor} makes this stream a natural fit. {base_blurb}"
    #elif score >= 4:
        #return f"Your relative comfort with {factor} helps here. {base_blurb}"
    #else:
        #return f"Among the traits that matter for this stream, {factor} was one of your higher areas — though it may still need development. {base_blurb}"

#def weakness_narrative(score, factor, base_blurb):
    #if score <= 3:
        #return f"Your low score in {factor} could make this stream more challenging. {base_blurb}"
    #elif score <= 6:
        #return f"{factor} may not be a major gap, but it’s an area to watch. {base_blurb}"
    #else:
        #return f"Even though {factor} is relatively strong overall, it ranked lowest here — so it’s worth attention. {base_blurb}"

def safe_text(val):
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()

def get_channel_narrative(channel_name, narratives, user_scores):
    """
    Select top 2 strengths and 1 weakness for this channel,
    then feed them into the AI blurb generator.
    """
    df = narratives[(narratives["channel_name"] == channel_name) & (narratives["weight"] >= 4)].copy()
    if df.empty:
        return f"No narrative data available for {channel_name}."

    # Map in user scores + weighted_score using factor_id
    df["factor_id"] = df["factor_name"].map(slugify)
    df["user_score"] = df["factor_id"].map(lambda fid: user_scores.get(fid, 0))
    df["weighted_score"] = df["user_score"] * df["weight"]

    # Edge case detection (per channel)
    all_high = df["user_score"].min() >= 8
    all_low = df["user_score"].max() <= 3

    # Select top 2 strengths
    strengths = df.sort_values("weighted_score", ascending=False).head(2)
    used_factors = strengths["factor_name"].tolist()

    # Select 1 weakest factor (not already used)
    weakness_candidates = df[~df["factor_name"].isin(used_factors)]
    if weakness_candidates.empty:
        return "Not enough factors to generate a weakness."
    weakness = weakness_candidates.sort_values("weighted_score", ascending=True).head(1)

    s1, s2, w1 = strengths.iloc[0], strengths.iloc[1], weakness.iloc[0]

    # --- Edge case handling ---
    if all_low:
        # ... build gentle reasons ...
        strengths_list = [s1["factor_name"], s2["factor_name"]]
        reasons = [
            f"Among your lower scores, {s1['factor_name']} still stood out as relatively stronger. {s1['strength_blurb']}",
            f"Similarly, {s2['factor_name']} showed some promise. {s2['strength_blurb']}",
            f"Your lowest area was {w1['factor_name']}, which could create challenges. {w1['weakness_blurb']}"
        ]
        return generate_channel_blurb(channel_name, strengths_list, w1["factor_name"], reasons)

    elif all_high:
        # ... build softened reasons ...
        strengths_list = [s1["factor_name"], s2["factor_name"]]
        reasons = [
            s1["strength_blurb"],
            s2["strength_blurb"],
        f"Even though your scores for {channel_name} are strong overall, {w1['factor_name']} still ranked lowest. {w1['weakness_blurb']}"
        ]
        return generate_channel_blurb(channel_name, strengths_list, w1["factor_name"], reasons)

    else:
     # Normal case
        strengths_list = [s1["factor_name"], s2["factor_name"]]
        reasons = [
            f"{s1['factor_name']}: {s1['strength_blurb']} This matters because {s1['weighting_reason']}.",
            f"{s2['factor_name']}: {s2['strength_blurb']} This matters because {s2['weighting_reason']}.",
            f"{w1['factor_name']}: {w1['weakness_blurb']} This could be a limitation because {w1['weighting_reason']}."
        ]
        return generate_channel_blurb(channel_name, strengths_list, w1["factor_name"], reasons)

st.set_page_config(page_title="Revenue Stream Compass — Top 5", page_icon="🌸", layout="centered")

BASE = Path(__file__).resolve().parent
XLSX = BASE / "Extreme_Weighting_Scoring_Prototype_for_FormWise_REPAIRED.xlsx"

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", str(s)).strip().lower()
    s = re.sub(r"[\s/]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s

def adjust_weight(w: float) -> float:
    """Remap Weights tab values (1–5) to nonlinear scale."""
    if pd.isna(w):
        return 0.0
    mapping = {1: 0.0, 2: 2.0, 3: 4.0, 4: 8.0, 5: 10.0}
    return mapping.get(int(w), 0.0)

@st.cache_data
def load_from_excel(xlsx_path: Path):
    if not xlsx_path.exists():
        st.error(f"Excel file not found:\n{xlsx_path}")
        st.stop()

    # --- Load Weights ---
    try:
        weights = pd.read_excel(xlsx_path, sheet_name="Weights").rename(columns=lambda c: str(c).strip())
    except Exception as e:
        st.error(f"Could not read 'Weights' sheet: {e}")
        st.stop()

    # --- Load Snippets (optional, already in your file) ---
    try:
        snippets = pd.read_excel(xlsx_path, sheet_name="Snippets").rename(columns=lambda c: str(c).strip())
    except Exception:
        snippets = pd.DataFrame(columns=["Revenue Stream","One-line Reason Template","Compass Chapter Link/Slug"])

    # --- Load Factor metadata + Categories ---
    try:
        factor_meta = pd.read_excel(xlsx_path, sheet_name="Factors").rename(columns=lambda c: str(c).strip())
        categories  = pd.read_excel(xlsx_path, sheet_name="Categories").rename(columns=lambda c: str(c).strip())
    except Exception:
        factor_meta = pd.DataFrame(columns=["factor_name","category_name","factor_description","left_label","right_label"])
        categories  = pd.DataFrame(columns=["category_name","category_description"])

    # --- Load Narratives ---
    try:
        narratives = pd.read_excel(
            xlsx_path,
            sheet_name="Narratives"
        ).rename(columns=lambda c: str(c).strip())
    except Exception as e:
        st.error(f"Could not read 'Narratives' sheet: {e}")
        narratives = pd.DataFrame(columns=["channel_name","factor_name","weight","strength_blurb","weakness_blurb"])

    # --- Build factors base (from Weights first column) ---
    first_col = weights.columns[0]
    factors = weights[[first_col]].dropna().drop_duplicates().copy()
    factors.columns = ["factor_name"]
    factors["factor_id"] = factors["factor_name"].map(slugify)
    factors["description"] = ""
    factors["min"] = 0
    factors["max"] = 10
    factors["step"] = 1

    # Merge in metadata from Factors sheet
    if not factor_meta.empty:
        factors = factors.merge(factor_meta, on="factor_name", how="left")

    # Always regenerate factor_id from factor_name to avoid KeyErrors
    factors["factor_id"] = factors["factor_name"].map(slugify)
    
    # --- Build channels ---
    w = weights.rename(columns={first_col: "factor"}).copy()
    channel_cols = [c for c in w.columns if c != "factor"]
    long = w.melt(id_vars=["factor"], value_vars=channel_cols,
                  var_name="channel_name", value_name="sensitivity_raw")
    if long["sensitivity_raw"].dropna().empty:
        st.error("No numeric sensitivities found in 'Weights'.")
        st.stop()

    long["factor_id"] = long["factor"].map(slugify)
    long["sensitivity"] = long["sensitivity_raw"].apply(adjust_weight)
    pivot = long.pivot_table(index="channel_name", columns="factor_id",
                             values="sensitivity", aggfunc="mean").reset_index()
    pivot.columns = ["channel_name"] + [f"f_{c}" for c in pivot.columns[1:]]
    pivot["channel_id"] = pivot["channel_name"].map(slugify)

    # Join snippets
    sn = snippets.rename(columns={
        "Revenue Stream": "channel_name",
        "One-line Reason Template": "why_fit_short",
        "Compass Chapter Link/Slug": "compass_link",
    })
    cols = [c for c in ["channel_name","why_fit_short","compass_link"] if c in sn.columns]
    sn = sn[cols] if cols else pd.DataFrame(columns=["channel_name","why_fit_short","compass_link"])
    channels = pivot.merge(sn, on="channel_name", how="left")
    for col in ["tags","caveats","startup_cost","labor_level","cashflow_speed"]:
        if col not in channels:
            channels[col] = ""

    factor_cols = [c for c in channels.columns if c.startswith("f_")]
    channels[factor_cols] = channels[factor_cols].fillna(0.0)

    return factors, categories, channels, narratives

#def test_api():
    #response = client.responses.create(
       # model="gpt-4.1-mini",
        #input="Say hello to Sharon in one short sentence."
    #)
    #return response.output_text

def generate_channel_blurb(channel, strengths, weakness, reasons):
    prompt = f"""
    Write a concise, authoritative explanation (3–4 sentences) about why this revenue stream is a fit
    for a flower farmer, grounded in the specific strengths, weaknesses, and reasons provided.

    - Channel: {channel}
    - Strength 1: {strengths[0]} → {reasons[0]}
    - Strength 2: {strengths[1]} → {reasons[1]}
    - Weakness: {weakness} → {reasons[2]}

    Guidelines:
    - Always speak directly to the farmer using "you" and "your" language.
    - Do not use "the farmer" or "a flower farmer" phrasing.- Use the blurbs and reasons directly — do not ignore or generalize them.
    - Use an authoritative, specific tone (avoid fluff).
    - Explain how the strengths lead to concrete outcomes (e.g., loyal buyers, upselling, efficiency).
    - Present the weakness clearly as a challenge to mitigate or plan for, but not making it sound too dire.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

# -------------------------
# APP STARTS
# -------------------------
st.title("Revenue Stream Compass™ — Quick Match")
st.caption("Rate your Field Factors to see your Top 5 revenue streams.")

factors, categories, channels, narratives = load_from_excel(XLSX)

# Safe default so any stray references won't crash before user clicks the button
rackstack = pd.DataFrame(columns=["channel_name", "score"])

# used when debugging and optimizing the math
# st.write("DEBUG: Factor names from Excel")
# st.json(factors["factor_name"].tolist())

# Debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode")

# -------------------------
# USER INPUTS
# -------------------------
user_scores = {}

st.subheader("Your Field Factor Self Assessment")

# factors, categories, channels now come from load_from_excel
for _, cat_row in categories.iterrows():
    cat_name = cat_row["category_name"]
    cat_desc = cat_row.get("category_description", "")

    st.markdown(f"## {cat_name}")
    if pd.notna(cat_desc) and str(cat_desc).strip():
        st.markdown(f"*{cat_desc.strip()}*")
        st.markdown("")

    # Filter factors belonging to this category
    these_factors = factors[factors["category_name"] == cat_name]

    for i, row in these_factors.iterrows():
        fid   = row["factor_id"]
        fname = row["factor_name"]

        # spacing before factors after the first
        if i > 0:
            st.markdown("&nbsp;", unsafe_allow_html=True)

        # Factor name + description
        fdesc = row.get("factor_description", "")
        st.markdown(f"**{fname}**: {safe_text(fdesc)}")

        # Custom labels, fallback to Weakness/Strength
        left_label  = safe_text(row.get("left_label"))  or "Weakness"
        right_label = safe_text(row.get("right_label")) or "Strength"

        # Slider
        vmin  = int(row.get("min", 0))
        vmax  = int(row.get("max", 10))
        vstep = int(row.get("step", 1))
        vdef  = int((vmax + vmin) // 2)

        user_scores[fid] = st.slider(
            fname,
            min_value=vmin,
            max_value=vmax,
            value=vdef,
            step=vstep,
            key=f"slider_{fid}",
            label_visibility="collapsed"
        )

        # Labels under slider
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; margin-top:-8px;'>"
            f"<span style='font-size:0.8em; color:gray;'>{left_label}</span>"
            f"<span style='font-size:0.8em; color:gray;'>{right_label}</span>"
            "</div>",
            unsafe_allow_html=True
        )

# -------------------------
# CALCULATE SCORES
# -------------------------
st.markdown("---")

# Button sets a flag in session_state
if st.button("See my Top 5"):
    st.session_state.show_results = True

# Only run calculations if flag is set
if st.session_state.get("show_results", False):
    factor_cols = [c for c in channels.columns if c.startswith("f_")]

    uw = {f"f_{fid}": float(user_scores[fid]) for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])

    uw_aligned = uw_df[channels[factor_cols].columns]

    # --- ORIGINAL SCORING ---
    # scores = np.dot(channels[factor_cols].values, uw_aligned.values.T).reshape(-1)
    #max_vector = np.array([10.0] * len(factor_cols))
    #max_scores = np.dot(channels[factor_cols].values, max_vector.T).reshape(-1)

    #ch = channels.copy()
    #ch["score"] = np.divide(scores, max_scores, out=np.zeros_like(scores), where=max_scores != 0)

    # --- NEW SCORING: Weighted Average with Coverage + Channel Normalization
    ch = channels.copy()
    max_factors = max(channels[factor_cols].astype(bool).sum(axis=1))

    adjusted_scores = []
    for idx, row in channels.iterrows():
        row_factors = row[factor_cols]
        factor_mask = row_factors > 0
        k = factor_mask.sum()

        if k == 0:
            adjusted_scores.append(0)
            continue

        factor_indices = row_factors.index[factor_mask]
        if len(factor_indices) == 0:
            adjusted_scores.append(0)
            continue

        scores = uw_aligned.loc[:, factor_indices].values.flatten()
        weights = row_factors[factor_mask].values

        if len(scores) == 0 or weights.sum() == 0:
            adjusted_scores.append(0)
            continue

        # Weighted average of user scores (0–10 scale)
        weighted_avg = np.dot(scores, weights) / weights.sum()

        # Normalize to 0–1
        normalized = weighted_avg / 10.0

        # Coverage adjustment (optional)
        coverage = k / max_factors

        score = normalized * coverage
        adjusted_scores.append(score)

    ch["score"] = adjusted_scores

    
    # --- Contribution Analysis (row-normalized) ---
    raw_contribs = channels[factor_cols].values * uw_aligned.values
    row_totals = raw_contribs.sum(axis=1, keepdims=True)
    contribs = np.divide(
        raw_contribs,
        row_totals,
        out=np.zeros_like(raw_contribs),
        where=row_totals != 0
    )
    contribs = pd.DataFrame(
        contribs,
        columns=factor_cols,
        index=channels["channel_name"]
    )
    score_map = dict(zip(channels["channel_name"], ch["score"]))
    contribs["normalized_total"] = contribs.index.map(score_map)
    
    def top_strengths_weaknesses(channel_name, n=2):
        row = contribs.loc[channel_name, factor_cols]
        sorted_factors = row.sort_values(ascending=False)
        strengths = sorted_factors.head(n).index.tolist()
        weaknesses = sorted_factors.tail(n).index.tolist()
        return strengths, weaknesses

    # Build Rack & Stack (all channels sorted by score)
    rackstack = (
        ch.loc[:, ["channel_name", "score"]]
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

    # --- Show Top 5 ---
    top5 = rackstack.head(5)
    st.subheader("Your Top 5 Matches")
    for _, r in top5.iterrows():
        with st.container(border=True):
            st.markdown(f"### {safe_text(r['channel_name'])}")
            st.markdown(f"**Score:** {r['score'] / 10:.0%}")

            chan = ch[ch["channel_name"] == r["channel_name"]].iloc[0]
            tags = safe_text(chan.get("tags"))
            if tags:
                st.markdown(f"**Tags:** {tags}")
            why = safe_text(chan.get("why_fit_short"))
            if why:
                st.markdown(f"_{why}_")
            link = safe_text(chan.get("compass_link"))
            if link:
                st.markdown(f"[Open Guidebook chapter]({link})")
    
    # --- Build portable Top 5 list for CTA ---
    top_5 = top5[["channel_name", "score"]].values.tolist()

    st.markdown("---")
    st.subheader("📩 Want to know *WHY* these are your Top 5 *AND* see how you stack up against all 18 potential revenue streams?")
    st.markdown(
        "Get a personalized explanation of your results delivered straight to your inbox — "
        "including some of the key strengths and challenges behind your Top 5 matches. We'll also "
        "give you your scores for all 18 possible revenue streams."
    )

    with st.form("email_capture"):
        first_name = st.text_input("First Name (required)")
        last_name = st.text_input("Last Name (optional)")
        farm_name = st.text_input("Farm Name (optional)")
        email = st.text_input("Email Address (required)")

        submitted = st.form_submit_button("Send Me My Report")

        if submitted and email and first_name:
            zapier_webhook_url = "https://hooks.zapier.com/hooks/catch/19897729/ud9fr8n/"
            payload = {
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "farm_name": farm_name,
                "top5": [c for c, s in top_5]
            }
            try:
                st.write("📡 Sending payload:", payload)   # Debug
                r = requests.post(zapier_webhook_url, json=payload)
                st.write("🔎 Response status:", r.status_code)  # Debug
                if r.status_code == 200:
                    st.success("✅ Thanks! Your personalized Top 5 explanation is on its way to your inbox.")
                else:
                    st.error(f"❌ Oops — something went wrong. Status {r.status_code}")
            except Exception as e:
                st.error(f"⚠️ Connection failed: {e}")

    st.info("✅ This block is DEV-only. This info will be put into a personalized pdf that gets sent via email.")

# -------------------------
# DEV/TEST OUTPUT (not shown in final lead magnet)
# -------------------------

#if st.button("Test AI"):
    #st.write(test_api())

st.markdown("---")
st.header("🌟 Your Top 5 Revenue Streams")

if 'top5' in locals() and not top5.empty:
    for _, r in top5.iterrows():
        channel = r["channel_name"]
        blurb = get_channel_narrative(channel, narratives, user_scores)
        st.markdown(f"### {safe_text(channel)}")
        st.markdown(f"**Score:** {r['score']:.0%}")
        st.write(blurb)

    # --- Full Rack & Stack (all channels, sorted) ---
    st.markdown("---")
    st.subheader("📊 Full Rack & Stack (All Channels)")

    rackstack_display = rackstack.copy()
    rackstack_display["Score"] = (rackstack_display["score"] * 100).round(1).astype(str) + "%"
    st.dataframe(rackstack_display[["channel_name", "Score"]])
else:
    st.info("👉 Click **See my Top 5** above to generate your personalized results.")

# -------------------------
# DEBUGGING STUFF
# -------------------------
    
    # 🚧 NOTE: Everything below is for internal dev/debug only.
    # 🚧 Do NOT include this section in the free/lead magnet version.

    #if debug_mode:
        #st.markdown("---")
        #st.markdown("### 🚧 DEV ONLY: Contribution & Rack & Stack 🚧")

        # Show contribution breakdown as percentages
        #st.markdown("#### Contribution Breakdown (as %)")
        #contribs_pct = contribs.copy()
        #factor_cols_only = [c for c in contribs.columns if c.startswith("f_")]

        # Convert factor columns to percentages
        #contribs_pct[factor_cols_only] = (contribs_pct[factor_cols_only] * 100).round(1)

        # Rename factor columns back to human-friendly names
        #factor_name_map = {
            #f"f_{row['factor_id']}": row['factor_name']
            #for _, row in factors.iterrows()
        #}
        #contribs_pct = contribs_pct.rename(columns=factor_name_map)

        # Convert normalized_total to %
        #contribs_pct["normalized_total"] = (contribs_pct["normalized_total"] * 100).round(1)

        #st.dataframe(contribs_pct)

        #st.markdown("#### All Channel Scores (Rack & Stack)")
        #st.dataframe(rackstack)

    # Debugging output removed for production
# if debug_mode:
#     if debug_mode:
        #raw_contribs = channels[factor_cols].values * uw_aligned.values
        #max_contribs = channels[factor_cols].values * 10.0
        #norm_contribs = np.divide(raw_contribs, max_contribs,
                                  #out=np.zeros_like(raw_contribs),
                                  #where=max_contribs != 0)

        #contribs = pd.DataFrame(norm_contribs, columns=factor_cols,
                                #index=channels["channel_name"])
        #score_map = dict(zip(channels["channel_name"], ch["score"]))
        #contribs["normalized_total"] = contribs.index.map(score_map)
        #contribs["normalized_score"] = contribs["normalized_total"]

        #st.markdown("### Debug: Contribution Breakdown (focus channels)")
        #st.dataframe(contribs.sort_values("normalized_score", ascending=False))

#adding this comment to test commit in VS Code

