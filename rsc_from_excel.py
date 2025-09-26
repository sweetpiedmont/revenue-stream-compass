import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests #needed for posting to Zapier
import os
from openai import OpenAI
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

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

def get_channel_short_narrative (channel_name, narratives, user_scores):
    """
    Generate the short narrative (~3–4 sentences) for lead magnet delivery.
    Handles normal + edge cases, then calls generate_channel_blurb().
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
    all_same = df["user_score"].nunique() == 1

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

    elif all_same:
        return (
        f"Because all of your Field Factor self-assessment scores were exactly the same, the explanation of your Top Revenue Streams is "
        f"based exclusively on how the Revenue Stream Compass™ weights different Field Factors in each revenue stream. "
        f"To get a more meaningful and personalized result, please consider adjusting your scores so they’re not all identical."
    )

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

    # Join snippets (only keep compass_link now)
    sn = snippets.rename(columns={
        "Revenue Stream": "channel_name",
        "Compass Chapter Link/Slug": "compass_link",
    })
    cols = [c for c in ["channel_name","compass_link"] if c in sn.columns]
    sn = sn[cols] if cols else pd.DataFrame(columns=["channel_name","compass_link"])
    channels = pivot.merge(sn, on="channel_name", how="left")

    factor_cols = [c for c in channels.columns if c.startswith("f_")]
    channels[factor_cols] = channels[factor_cols].fillna(0.0)

    return factors, categories, channels, narratives, factor_to_color

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

# ------------------------
# LONG NARRATIVES
# ------------------------
def get_channel_long_narrative(channel_name, narratives, user_scores, compass_link=None):
    """
    Generate the long narrative (~2 short paragraphs) for paid Compass customers.
    - Uses ALL factors with weight >= 4.
    - Groups into strengths (score >= 7), weaknesses (score <= 3).
    - If no strengths, will borrow from neutral factors (scores 4–6),
      but clearly labeled as 'relative strengths'.
    - Includes optional Guidebook/Compass chapter link at the end if provided.
    """

    # Filter to factors for this channel with weight >= 4
    df = narratives[(narratives["channel_name"] == channel_name) & (narratives["weight"] >= 4)].copy()
    if df.empty:
        return f"No narrative data available for {channel_name}."

    # Add user scores
    df["factor_id"] = df["factor_name"].map(slugify)
    df["user_score"] = df["factor_id"].map(lambda fid: user_scores.get(fid, 0))

    # Split into groups
    strengths = df[df["user_score"] >= 7]
    weaknesses = df[df["user_score"] <= 3]
    neutrals   = df[(df["user_score"] >= 4) & (df["user_score"] <= 6)]

    # Handle case where no true strengths exist
    if strengths.empty and not neutrals.empty:
        strengths = neutrals.sort_values("user_score", ascending=False).head(2)
        borrowed = True
    else:
        borrowed = False

    # Convert to list of dicts
    strengths_list = strengths.to_dict("records")
    weaknesses_list = weaknesses.to_dict("records")

    # Call the generator
    long_blurb = generate_channel_long_blurb(
        channel=channel_name,
        strengths=strengths_list,
        weaknesses=weaknesses_list,
        borrowed=borrowed
    )

    # Add Guidebook link if available
    if compass_link:
        long_blurb += f"\n\n📖 Read more in your Compass Guidebook: {compass_link}"

    return long_blurb


def generate_channel_long_blurb(channel, strengths, weaknesses, borrowed=False):
    """
    Utility for long narratives.
    Takes lists of strengths and weaknesses, builds the AI prompt,
    and returns a ~2-paragraph narrative.
    """

    strengths_text = "\n".join([
        f"- {s['factor_name']}: {s['strength_blurb']}" for s in strengths
    ]) if strengths else "None"

    weaknesses_text = "\n".join([
        f"- {w['factor_name']}: {w['weakness_blurb']}" for w in weaknesses
    ]) if weaknesses else "None"

    borrowed_note = (
        "Note: These aren’t true strengths — they are simply the relatively stronger factors "
        "from your self-assessment that still play a role for this revenue stream."
        if borrowed else ""
    )

    prompt = f"""
    Write a detailed, authoritative narrative for a flower farmer about their fit for this revenue stream.

    Channel: {channel}

    Strengths:
    {strengths_text}

    Weaknesses:
    {weaknesses_text}

    Guidelines:
    - Write 2 short paragraphs (6–10 sentences total).
    - Use direct "you/your" language.
    - Clearly explain how the strengths contribute to success in this stream.
    - Clearly explain how the weaknesses might create challenges.
    - If strengths were borrowed (see below), make that distinction explicit.
    - Tone: supportive, expert, and practical (not fluffy).
    - Do NOT suggest specific fixes or workarounds here (those live in the Guidebook).

    Borrowed strengths note: {borrowed_note}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

def render_navigation_page(channel_name, narrative, advantages, obstacles, rank, compass_link=None):
    # Set up Jinja environment
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("revenue_stream_page.html")

    # Render HTML with actual values
    html = template.render(
        RevenueStreamName=channel_name,
        Narrative=narrative,
        AdvantagesList=advantages,   # pass as list
        ObstaclesList=obstacles,     # pass as list
        Rank=rank,
        CompassLink=compass_link or ""
    )

    return html

    pages_html = []

    for rank, row in enumerate(ranked_channels.itertuples(), start=1):
        channel = row.channel_name
        score = row.score
        narrative = narratives.get(channel, "⚠️ No narrative available.")

        # --- Build factor highlights ---
        df = narratives_df[(narratives_df["channel_name"] == channel) & (narratives_df["weight"] >= 4)].copy()
        df["factor_id"] = df["factor_name"].map(slugify)
        df["user_score"] = df["factor_id"].map(lambda fid: user_scores.get(fid, 0))

        advantages = df[df["user_score"] >= 7]["factor_name"].tolist()
        obstacles  = df[df["user_score"] <= 3]["factor_name"].tolist()

        # If no strong advantages, borrow the “least weak” factors
        if not advantages and not df.empty:
            borrowed = df.sort_values("user_score", ascending=False).head(2)["factor_name"].tolist()
            advantages = [f"(Relative) {x}" for x in borrowed]

        data = {
            "RevenueStreamName": channel,
            "Narrative": narrative,
            "Rank": rank,
            "Compatibility": f"{score:.0%}",  # Example: 87%
            "AdvantagesList": "".join([f"<div class='tag green'>{a}</div>" for a in advantages]),
            "ObstaclesList": "".join([f"<div class='tag red'>{o}</div>" for o in obstacles]),
        }

        page_html = render_navigation_page(data)
        pages_html.append(page_html)

    # Combine all pages, separated by page breaks
    return "<div style='page-break-after: always;'></div>".join(pages_html)

# -------------------------
# APP STARTS
# -------------------------

factors, categories, channels, narratives = load_from_excel(XLSX)

# Build factor → category color map
factor_to_category = dict(zip(factor_meta["factor_name"], factor_meta["category_name"]))
category_to_color = dict(zip(categories["category_name"], categories["category_color"]))

# Factor to color map (using category link)
factor_to_color = {f: category_to_color.get(cat, "#cccccc") 
                   for f, cat in factor_to_category.items()}


# Safe default so any stray references won't crash before user clicks the button
rackstack = pd.DataFrame(columns=["channel_name", "score"])

# -------------------------
# PHASE 0: INTRO + TRUST SETUP
# -------------------------

if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    # Pre-amble
    st.title("🌸 Welcome to Your Field Factors Self Assessment")
    st.markdown("""
    This short, structured quiz is designed to show which flower-farming revenue streams align with your strengths and resources.  
    """)

    # Honesty reminder
    st.subheader("A quick but important note before you begin")
    st.markdown("""
    👉 There are no “good” or “bad” answers. The more honest you are, the more useful your results will be. 
    
    It may be tempting to nudge your scores toward the revenue stream you think you want to pursue — or 
    soften your answers because you’re worried a low score means you can’t pursue it.  
    
    That’s not how this works.  

    This process isn’t here to gatekeep your dream. It’s here to help you pursue it smarter. Even if you've got low scores in critical areas, we're here to help you find workarounds.  
    
    So be honest: don’t downplay your strengths, and don’t be afraid to reveal your challenges.
    """)

    # Boxed callout
    st.markdown("---")
    st.markdown(
        """
        <div style='border: 2px solid #ddd; border-radius: 10px; padding: 15px; background-color: #f9f9f9;'>
        <strong>Note: Not Every Factor Is Scored — On Purpose</strong><br><br>
        You won’t see questions here about things like years of farming experience, bookkeeping skills, 
        or how many other farms are nearby. Those matter — but they’ll affect you no matter which 
        sales channel you choose.  

        This self-assessment focuses only on the factors that actually help you compare and choose 
        between different sales channels. If it’s not here, it’s not because it’s unimportant — 
        it’s because it won’t change which options are the best fit for you.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Privacy + Consent
    consent = st.checkbox("🔒 I understand that my responses will be stored securely, "
                          "used only in aggregate for research, and never shared individually.")
    
    # Start button (only active if consent given)
    if st.button("🌟 I'm Ready — Let's Start My Self-Assessment!", disabled=not consent):
        if consent:
            st.session_state.started = True
            st.rerun()

else:
    # -------------------------
    # USER INPUTS
    # -------------------------
    user_scores = {}

    st.title("🌸 Welcome to Your Field Factors Self Assessment")

    st.markdown("👉 For each question, use the slider to rate yourself on the scale provided. Move the bar to the point that best reflects your current situation.")

    for _, cat_row in categories.iterrows():
        cat_name = cat_row["category_name"]
        cat_desc = cat_row.get("category_description", "")

        st.markdown(f"## {cat_name}")
        if pd.notna(cat_desc) and str(cat_desc).strip():
            st.markdown(f"*{cat_desc.strip()}*")
            st.markdown("")

        these_factors = factors[factors["category_name"] == cat_name]

        for i, row in these_factors.iterrows():
            fid   = row["factor_id"]
            fname = row["factor_name"]

            if i > 0:
                st.markdown("&nbsp;", unsafe_allow_html=True)

            fdesc = row.get("factor_description", "")
            st.markdown(f"**{fname}**: {safe_text(fdesc)}")

            left_label  = safe_text(row.get("left_label"))  or "Weakness"
            right_label = safe_text(row.get("right_label")) or "Strength"

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

            st.markdown(
                f"<div style='display:flex; justify-content:space-between; margin-top:-8px;'>"
                f"<span style='font-size:0.8em; color:gray;'>{left_label}</span>"
                f"<span style='font-size:0.8em; color:gray;'>{right_label}</span>"
                "</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    if st.button("See My Top 5"):
        st.session_state.show_results = True

# -------------------------
# CALCULATE SCORES
# -------------------------

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

        # Weighted average of user scores (0–10 scale)
        weighted_avg = np.dot(scores, weights) / weights.sum()

        # Normalize to 0–1
        score = weighted_avg / 10.0

        adjusted_scores.append(score)

    # Existing: fit-only score
    ch["fit_score"] = adjusted_scores  # already normalized 0–1

    # Option 2: Weighted blend (70% fit + 30% coverage)
    max_factors = max(channels[factor_cols].astype(bool).sum(axis=1))
    
    # --- Coverage calculation (count only "important" factors: Excel 4–5 → adjusted 8–10) ---
    max_factors_high = (channels[factor_cols] >= 8).sum(axis=1).max()

    coverages = []
    for idx, row in channels.iterrows():
        k = (row[factor_cols] >= 8).sum() # count only factors with adjusted weight >= 8
        coverage = k / max_factors_high if max_factors_high > 0 else 0
        coverages.append(coverage)

    ch["coverage"] = coverages

    # --- Blended scores ---
    ch["blend_score_70"] = 0.7 * ch["fit_score"] + 0.3 * ch["coverage"]
    ch["score"] = ch["blend_score_70"]  # Lock in 70/30 as the scoring method
    #ch["blend_score_80"] = 0.8 * ch["fit_score"] + 0.2 * ch["coverage"] #remove this option

    ### REMOVED RADIO BUTTON AND OPTIONS FOR HOW SCORING IS DONE ###
    # Let user choose which scoring method drives the rankings
    #score_method = st.radio(
        #"Choose scoring method for Top 5:",
        #["Fit Only", "Blend (70/30)", "Blend (80/20)", "Coverage Only"],
        #index=1   # default = Blend (70/30)
    #)

    #if score_method == "Fit Only":
        #ch["score"] = ch["fit_score"]
    #elif score_method == "Coverage Only":
        #ch["score"] = ch["coverage"]
    #elif score_method == "Blend (70/30)":
        #ch["score"] = ch["blend_score_70"]
    #elif score_method == "Blend (80/20)":
        #ch["score"] = ch["blend_score_80"]
    # else:
        #ch["score"] = ch["fit_score"]  # fallback

    # Keep score_map synced
    score_map = dict(zip(ch["channel_name"], ch["score"]))
    
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

    rackstack = (
        ch.loc[:, ["channel_name", "fit_score", "coverage", "blend_score_70", "score"]]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
    )

    # --- Build Navigation Planner pages (all 18 streams, ordered by rank) ---
    # --- Build long narratives for ALL 18 channels (paid Compass only) ---
    long_narratives = {}
    for channel in ch["channel_name"].unique():
        slug = ch.loc[ch["channel_name"] == channel, "compass_link"].values[0]
        long_narratives[channel] = get_channel_long_narrative(
            channel,
            narratives,
            user_scores,
        )

    planner_pages = []
    for i, row in rackstack.iterrows():
        ch_name = row["channel_name"]
        rank = i + 1
        narrative = long_narratives.get(ch_name, "")

        # Get high/low user factors for this channel
        df = narratives[(narratives["channel_name"] == ch_name) & (narratives["weight"] >= 4)].copy()
        df["factor_id"] = df["factor_name"].map(slugify)
        df["user_score"] = df["factor_id"].map(lambda fid: user_scores.get(fid, 0))

        advantages = df[df["user_score"] >= 7]["factor_name"].tolist()
        obstacles  = df[df["user_score"] <= 3]["factor_name"].tolist()

        compass_link = ch.loc[ch["channel_name"] == ch_name, "compass_link"].values[0]

        # Render HTML for this page
        page_html = render_navigation_page(
            channel_name=ch_name,
            narrative=narrative,
            advantages=advantages,
            obstacles=obstacles,
            rank=rank,
            compass_link=compass_link,
        )
        planner_pages.append(page_html)

    # --- Show Top 5 ---
    # Global check: did the user leave all sliders the same?
    if len(set(user_scores.values())) == 1:
        st.warning(
            "⚠️ It looks like you gave every Field Factor the same score. "
            "Your results will be based only on how the Compass weights different Field Factors, not your unique situation. "
            "For a more meaningful result, try adjusting your scores so they’re not all identical."
        )

    top5 = rackstack.head(5)
    st.subheader("Your Top 5 Matches")

    for i, r in enumerate(top5.itertuples(), start=1):
        st.markdown(f"#### {i}. {safe_text(r.channel_name)}")
    
    # --- Build portable Top 5 list for CTA ---
    top_5 = top5[["channel_name", "score"]].values.tolist()

    st.markdown("---")
    st.subheader("📩 Want to know *WHY* these are your Top 5 *AND* see how you stack up against all 18 potential revenue streams?")
    st.markdown(
        "Get a personalized explanation of your Top 5 revenue stream results delivered straight to your inbox — "
        "including some of the key strengths and challenges behind your Top 5 matches. We'll also "
        "show you how you fit against all 18 possible revenue streams."
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

st.write("DEBUG: Sample entries", list(long_narratives.items())[:3])

result = get_channel_long_narrative(channel, narratives, user_scores, compass_link=slug)
if not result:
    result = f"(⚠️ No narrative generated for {channel})"
long_narratives[channel] = result

# -------------------------
# DEV/TEST OUTPUT (not shown in final lead magnet)
# -------------------------

st.markdown("---")
st.header("🌟 Your Top 5 Revenue Streams")

if 'top5' in locals() and not top5.empty:
    for _, r in top5.iterrows():
        channel = r["channel_name"]
        short_narrative = get_channel_short_narrative(channel, narratives, user_scores)
        st.markdown(f"### {safe_text(channel)}")
        st.markdown(f"**Score:** {r['score']:.0%}")
        st.write(short_narrative)

    # --- Full Rack & Stack (all channels, sorted) ---
    st.markdown("---")
    st.subheader("📊 Full Rack & Stack (All Channels)")

    rackstack_display = rackstack.copy()
    rackstack_display["Score"] = (rackstack_display["score"] * 100).round(1).astype(str) + "%"
    st.dataframe(rackstack_display[["channel_name", "Score"]])
else:
    st.info("👉 Click **See my Top 5** above to generate your personalized results.")

# 🚧 DEV-ONLY: Preview a couple long narratives
with st.expander("🚧 DEV ONLY: Preview Long Narratives for Paid Compass"):
    preview_channels = ["Community and Corporate Events", "Workshops", "Wholesaling via a Collective"]
    for ch_name in preview_channels:
        st.markdown(f"### {ch_name}")
        st.write(long_narratives[ch_name])
        st.markdown("---")

with st.expander("🚧 DEV ONLY: Preview Navigation Planner"):
    for page in planner_pages[:3]:  # just show first 3 pages for testing
        st.components.v1.html(page, height=500, scrolling=True)
        st.markdown("---")
