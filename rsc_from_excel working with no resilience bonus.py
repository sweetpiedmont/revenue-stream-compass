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
    try:
        weights = pd.read_excel(xlsx_path, sheet_name="Weights").rename(columns=lambda c: str(c).strip())
    except Exception as e:
        st.error(f"Could not read 'Weights' sheet: {e}")
        st.stop()
    try:
        snippets = pd.read_excel(xlsx_path, sheet_name="Snippets").rename(columns=lambda c: str(c).strip())
    except Exception:
        snippets = pd.DataFrame(columns=["Revenue Stream","One-line Reason Template","Compass Chapter Link/Slug"])

    # Build factors
    first_col = weights.columns[0]
    factors = weights[[first_col]].dropna().drop_duplicates().copy()
    factors.columns = ["factor_name"]
    factors["factor_id"] = factors["factor_name"].map(slugify)
    factors["description"] = ""
    factors["min"] = 0
    factors["max"] = 10
    factors["step"] = 1

    # Build channels by melting -> normalizing -> pivoting
    w = weights.rename(columns={first_col: "factor"}).copy()
    channel_cols = [c for c in w.columns if c != "factor"]
    long = w.melt(id_vars=["factor"], value_vars=channel_cols, var_name="channel_name", value_name="sensitivity_raw")
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

    return factors, channels

# -------------------------
# APP STARTS
# -------------------------
st.title("Revenue Stream Compass™ — Quick Match")
st.caption("Rate your Field Factors to see your Top 3 revenue streams.")

factors, channels = load_from_excel(XLSX)

# Debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode")

# -------------------------
# DEBUG HARNESS
# -------------------------
if debug_mode:
    st.markdown("## 🔍 Debug Harness")

    # 1. Show factor columns (slugs)
    st.write("Factor columns in channels:")
    st.json([c for c in channels.columns if c.startswith("f_")])

    # 2. Show Weddings row for all factors
    wedding_rows = channels[channels["channel_name"].str.contains("Wedding", case=False)]
    st.write("Weights (after mapping) for Wedding-related channels:")
    st.dataframe(wedding_rows)

    # 3. Show user slider vector
    if 'user_scores' in locals():
        st.write("User slider vector (uw):")
        uw = {f"f_{fid}": float(user_scores[fid]) for fid in user_scores.keys()}
        st.json(uw)

        # Match Weddings vs user sliders for Customer Service
        cs_col = [c for c in uw.keys() if "customer_service" in c or "sales" in c]
        if cs_col:
            st.write("Customer Service & Sales factor detected:", cs_col)
            for _, row in wedding_rows.iterrows():
                st.write(f"{row['channel_name']} → weight={row[cs_col[0]]}, "
                         f"user={uw[cs_col[0]]}, "
                         f"contrib={row[cs_col[0]] * uw[cs_col[0]]}")
                
scenario = None
chosen_factor = None
if debug_mode:
    scenario = st.selectbox(
        "Choose a test scenario:",
        [
            "All = 5",
            "All = 10",
            "One factor = 10, rest = 0",
            "All = 10 except one = 0",
        ]
    )
    if scenario in ["One factor = 10, rest = 0", "All = 10 except one = 0"]:
        chosen_factor = st.selectbox("Select factor:", factors["factor_name"].tolist())

# -------------------------
# USER INPUTS
# -------------------------
user_scores = {}

if debug_mode and scenario:
    for _, row in factors.iterrows():
        fid = row["factor_id"]
        fname = row["factor_name"]

        if scenario == "All = 5":
            user_scores[fid] = 5
        elif scenario == "All = 10":
            user_scores[fid] = 10
        elif scenario == "One factor = 10, rest = 0":
            user_scores[fid] = 10 if fname == chosen_factor else 0
        elif scenario == "All = 10 except one = 0":
            user_scores[fid] = 0 if fname == chosen_factor else 10
else:
    st.subheader("Your Field Factor Scores")
    cols = st.columns(2)
    for i, row in factors.iterrows():
        col = cols[i % 2]
        user_scores[row["factor_id"]] = col.slider(
            row["factor_name"],
            min_value=int(row["min"]),
            max_value=int(row["max"]),
            value=int((row["max"] + row["min"]) // 2),
            step=int(row["step"]),
            help=row.get("description", "")
        )

# -------------------------
# CALCULATE
# -------------------------
st.markdown("---")
if st.button("See my Top 3"):
    factor_cols = [c for c in channels.columns if c.startswith("f_")]

    # User self-scores (0–10 from sliders)
    uw = {f"f_{fid}": float(user_scores[fid]) for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])

    # Safety check
    missing = sorted(set(uw_df.columns) - set(factor_cols))
    if missing:
        st.error(f"These factors are missing in Weights → {missing}")
        st.stop()

    # -------------------------
    # SCORING
    # -------------------------
    # Raw = Σ(weight × user_slider), force column alignment
    uw_aligned = uw_df[channels[factor_cols].columns]
    scores = np.dot(channels[factor_cols].values, uw_aligned.values.T).reshape(-1)

    # Max possible = Σ(weight × 10)  (fixed, not based on user sliders)
    max_vector = np.array([10.0] * len(factor_cols))
    max_scores = np.dot(channels[factor_cols].values, max_vector.T).reshape(-1)

    # Normalized score = raw / max_possible
    ch = channels.copy()
    ch["score"] = np.divide(scores, max_scores, out=np.zeros_like(scores), where=max_scores != 0)

    # -------------------------
    # DISPLAY
    # -------------------------
    rackstack = ch[["channel_name", "score"]].sort_values("score", ascending=False)
    st.markdown("### All Channel Scores (Rack & Stack)")
    st.dataframe(rackstack)

    # Top 3
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

    # Debug tables
    if debug_mode:
        # Raw contributions (factor × user slider)
        raw_contribs = channels[factor_cols].values * uw_df.values

        # Max contributions (factor × 10)
        max_contribs = channels[factor_cols].values * 10.0

        # Normalize each factor’s contribution (0–1 scale) against its max possible
        norm_contribs = np.divide(
            raw_contribs,
            max_contribs,
            out=np.zeros_like(raw_contribs),
            where=max_contribs != 0
        )

        contribs = pd.DataFrame(
            norm_contribs,
            columns=factor_cols,
            index=channels["channel_name"]
        )

        # Bring in main normalized channel scores
        score_map = dict(zip(channels["channel_name"], ch["score"]))
        contribs["normalized_total"] = contribs.index.map(score_map)
        contribs["normalized_score"] = contribs["normalized_total"]

        # -------------------------
        # Extra debug for Custom Weddings
        # -------------------------
        test_channel = "Custom Weddings"
        if test_channel in ch["channel_name"].values:
            # Use positional index, not index label
            pos = ch.index.get_loc(ch[ch["channel_name"] == test_channel].index[0])

            st.write(f"Raw total score = {scores[pos]}")
            st.write(f"Max possible score = {max_scores[pos]}")
            st.write(f"Normalized score (raw/max) = {ch.iloc[pos]['score']:.4f}")

            st.write("Factor × User Slider contributions:")
            for col in factor_cols:
                w = channels.iloc[pos][col]
                u = uw_df[col].iloc[0]
                st.write(f"{col}: weight={w}, user={u}, contrib={w*u}")

        # Row-wise % contribution breakdown
        contribs_pct = contribs[factor_cols].div(contribs[factor_cols].sum(axis=1), axis=0).fillna(0) * 100
        contribs_pct["normalized_score"] = contribs["normalized_score"]

        # Prettify column labels (remove "f_" prefix)
        contribs_display = contribs.rename(columns=lambda c: c.replace("f_", ""))
        contribs_pct_display = contribs_pct.rename(columns=lambda c: c.replace("f_", ""))

        focus_channels = ["Custom Weddings", "A la Carte Weddings", "W/S to Local Florists", "W/S to Wholesalers"]
        st.markdown("### Debug: Contribution Breakdown (focus channels)")
        st.dataframe(
            contribs_display.loc[contribs_display.index.intersection(focus_channels)]
            .sort_values("normalized_score", ascending=False)
        )
        st.markdown("### Debug: % Contribution by Factor (focus channels)")
        st.dataframe(
            contribs_pct_display.loc[contribs_pct_display.index.intersection(focus_channels)]
            .sort_values("normalized_score", ascending=False)
        )

        with st.expander("See full contribution table"):
            st.dataframe(contribs_display.sort_values("normalized_score", ascending=False))

