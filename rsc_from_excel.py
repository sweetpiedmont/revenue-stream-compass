import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests #needed for posting to Zapier

def strength_narrative(score, factor, base_blurb):
    if score >= 7:
        return f"Your strong {factor} makes this stream a natural fit. {base_blurb}"
    elif score >= 4:
        return f"Your relative comfort with {factor} helps here. {base_blurb}"
    else:
        return f"Among the traits that matter for this stream, {factor} was one of your higher areas — though it may still need development. {base_blurb}"

def weakness_narrative(score, factor, base_blurb):
    if score <= 3:
        return f"Your low score in {factor} could make this stream more challenging. {base_blurb}"
    elif score <= 6:
        return f"{factor} may not be a major gap, but it’s an area to watch. {base_blurb}"
    else:
        return f"Even though {factor} is relatively strong overall, it ranked lowest here — so it’s worth attention. {base_blurb}"

def safe_text(val):
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
# CALCULATE SCORES
# -------------------------
st.markdown("---")

# Button sets a flag in session_state
if st.button("See my Top 3"):
    st.session_state.show_results = True

# Only run calculations if flag is set
if st.session_state.get("show_results", False):
    factor_cols = [c for c in channels.columns if c.startswith("f_")]

    uw = {f"f_{fid}": float(user_scores[fid]) for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])

    uw_aligned = uw_df[channels[factor_cols].columns]

    scores = np.dot(channels[factor_cols].values, uw_aligned.values.T).reshape(-1)
    max_vector = np.array([10.0] * len(factor_cols))
    max_scores = np.dot(channels[factor_cols].values, max_vector.T).reshape(-1)

    ch = channels.copy()
    ch["score"] = np.divide(scores, max_scores, out=np.zeros_like(scores), where=max_scores != 0)

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

    # --- Narrative function (insert here) ---
    def get_channel_narrative(channel_name, contribs, narratives, user_scores, used_factors):
        row = contribs.loc[channel_name]
        narr = narratives[narratives["channel_name"] == channel_name]
        narr = narr[narr["weight"] >= 4]

        df = pd.DataFrame({
            "factor": row.index,
            "contribution": row.values,
            "user_score": [user_scores.get(f, 0) for f in row.index]
        }).merge(narr, left_on="factor", right_on="factor_name", how="inner")

        strengths = []
        for _, subdf in df.sort_values("contribution", ascending=False).iterrows():
            if subdf["factor"] not in used_factors:
                s_text = strength_narrative(subdf["user_score"], subdf["factor_name"], subdf["strength_blurb"])
                strengths.append(s_text)
                used_factors.add(subdf["factor"])
            if len(strengths) == 2:
                break

        weakest = df.sort_values("contribution", ascending=True).iloc[0]
        w_text = weakness_narrative(weakest["user_score"], weakest["factor_name"], weakest["weakness_blurb"])

        return {
            "channel": channel_name,
            "strengths": strengths,
            "weakness": w_text
        }
    
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

    # --- Show Top 3 ---
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

    # --- Build portable Top 3 list for CTA ---
    top_3 = top3[["channel_name", "score"]].values.tolist()

    st.markdown("---")
    st.subheader("📩 Want to Know *Why* These Are Your Top 3?")
    st.markdown(
        "Get a personalized explanation of your results delivered straight to your inbox — "
        "including some of the key strengths and challenges behind your Top 3 matches."
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
                "top3": [c for c, s in top_3]
            }
            try:
                st.write("📡 Sending payload:", payload)   # Debug
                r = requests.post(zapier_webhook_url, json=payload)
                st.write("🔎 Response status:", r.status_code)  # Debug
                if r.status_code == 200:
                    st.success("✅ Thanks! Your personalized Top 3 explanation is on its way to your inbox.")
                else:
                    st.error(f"❌ Oops — something went wrong. Status {r.status_code}")
            except Exception as e:
                st.error(f"⚠️ Connection failed: {e}")

# ----
# testing minimal narrative block
#------

# --- DEV/TEST: Load narratives from Excel ---
narratives = pd.read_excel(
    "Extreme_Weighting_Scoring_Prototype_for_FormWise_REPAIRED.xlsx",
    sheet_name="Narratives"
)

st.header("🧪 Narrative Test Block (DEV)")

# Example Top 3 channel from scoring (hardcoded for now)
top_channel = "Farmers Market"

st.subheader(f"Testing narratives for: {top_channel}")

# Filter Narratives sheet
subset = narratives[
    (narratives["channel_name"] == top_channel) & 
    (narratives["weight"] >= 4)
]

# Show first couple blurbs
for _, row in subset.head(2).iterrows():
    st.write(f"**Strength:** {row['strength_blurb']}")
    st.write(f"**Weakness:** {row['weakness_blurb']}")
    st.caption(f"Reason: {row['weighting_reason']}")
    st.markdown("---")

st.info("✅ This block is DEV-only. It won’t run in main until you merge it back.")


# -------------------------
# DEV/TEST OUTPUT (not shown in final lead magnet)
# -------------------------
#st.markdown("## 🔧 Developer/Test Output: Narrative Blurbs")
#st.caption("This section is only for testing the new blurb logic. It will not appear in the final user-facing app.")

#used_factors = set()

#for _, r in top3.iterrows():
    #channel = r["channel_name"]
    #narrative = get_channel_narrative(channel, contribs, narratives, user_scores, used_factors)

    #st.subheader(narrative["channel"])
    #for s in narrative["strengths"]:
        #st.write("🌟", s)
    #st.write("⚠️", narrative["weakness"])


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
        c#ontribs_pct["normalized_total"] = (contribs_pct["normalized_total"] * 100).round(1)

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

