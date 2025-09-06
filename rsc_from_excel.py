import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re

def safe_text(val):
    import pandas as pd
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()

st.set_page_config(page_title="Revenue Stream Compass — Top 3", page_icon="🌸", layout="centered")

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

    return factors, categories, channels

# -------------------------
# APP STARTS
# -------------------------
st.title("Revenue Stream Compass™ — Quick Match")
st.caption("Rate your Field Factors to see your Top 3 revenue streams.")

factors, categories, channels = load_from_excel(XLSX)

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
# CALCULATE
# -------------------------
st.markdown("---")
if st.button("See my Top 3"):
    factor_cols = [c for c in channels.columns if c.startswith("f_")]

    uw = {f"f_{fid}": float(user_scores[fid]) for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])

    # 🚨 TEMPORARY PATCH 🚨
    # At the moment, only a subset of Field Factors (e.g., 2 sliders) are being rendered in the UI.
    # That means uw_df is missing many factor columns that exist in channels[factor_cols].
    # This reindex step forces uw_df to have ALL factor columns, filling missing ones with 0.
    # ⚠️ IMPORTANT: Once the app is updated to render ALL Field Factors with sliders (with proper
    # descriptions + labels pulled from Excel), this reindex line (below) should be REMOVED.
    uw_df = uw_df.reindex(columns=channels[factor_cols].columns, fill_value=0)

    uw_aligned = uw_df[channels[factor_cols].columns]


    scores = np.dot(channels[factor_cols].values, uw_aligned.values.T).reshape(-1)
    max_vector = np.array([10.0] * len(factor_cols))
    max_scores = np.dot(channels[factor_cols].values, max_vector.T).reshape(-1)

    ch = channels.copy()
    ch["score"] = np.divide(scores, max_scores, out=np.zeros_like(scores), where=max_scores != 0)

    # --- Contribution Analysis ---
    raw_contribs = channels[factor_cols].values * uw_aligned.values
    max_contribs = channels[factor_cols].values * 10.0

    # Normalize by factor importance (avoid divide by zero)
    norm_contribs = np.divide(
        raw_contribs, 
        max_contribs, 
        out=np.zeros_like(raw_contribs), 
        where=max_contribs != 0
    )

    # Make a DataFrame of contributions per factor per channel
    contribs = pd.DataFrame(
        norm_contribs, 
        columns=factor_cols, 
        index=channels["channel_name"]
)

    # Add total score column for easy sorting
    score_map = dict(zip(channels["channel_name"], ch["score"]))
    contribs["normalized_total"] = contribs.index.map(score_map)

    # Example: get top strengths and weaknesses for a channel
    def top_strengths_weaknesses(channel_name, n=2):
        row = contribs.loc[channel_name, factor_cols]
        sorted_factors = row.sort_values(ascending=False)
        strengths = sorted_factors.head(n).index.tolist()
        weaknesses = sorted_factors.tail(n).index.tolist()
        return strengths, weaknesses

    # Build Rack & Stack (all channels sorted by score, highest first)
    rackstack = (
        ch.loc[:, ["channel_name", "score"]]
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

    # --- Show Top 3 first ---
    top3 = rackstack.head(3)
    st.subheader("Top 3 Matches")
    for _, r in top3.iterrows():
        with st.container(border=True):
            st.markdown(f"### {safe_text(r['channel_name'])}")
            st.markdown(f"**Score:** {r['score']:.2%}")

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

    # 🚧 NOTE: Everything below is for internal dev/debug only.
    # 🚧 Do NOT include this section in the free/lead magnet version.

    if debug_mode:
        st.markdown("---")
        st.markdown("### 🚧 DEV ONLY: Contribution & Rack & Stack 🚧")

        st.markdown("#### Contribution Breakdown")
        st.dataframe(contribs)

        st.markdown("#### All Channel Scores (Rack & Stack)")
        st.dataframe(rackstack)

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

