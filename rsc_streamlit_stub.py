
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Revenue Stream Compass — Top 3", page_icon="🌸", layout="centered")

st.title("Revenue Stream Compass™ — Quick Match")
st.caption("Rate your Field Factors to see your Top 3 revenue streams.")

@st.cache_data
def load_data():
    factors = pd.read_csv("factors.csv")
    channels = pd.read_csv("channels.csv")
    return factors, channels

factors, channels = load_data()

# Build input UI dynamically from factors.csv
st.subheader("Your Field Factor Scores")
user_scores = {}
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

st.markdown("---")
if st.button("See my Top 3"):
    # Prepare scoring
    # Normalize weights equally for MVP (you can extend to real weights later)
    weights = {fid: 1.0 for fid in user_scores.keys()}
    w_sum = sum(weights.values())
    weights = {k: v / w_sum for k, v in weights.items()}

    # Build a DataFrame of user * weight for each factor
    uw = {f"f_{fid}": user_scores[fid] * weights[fid] for fid in user_scores.keys()}
    uw_df = pd.DataFrame([uw])

    # Compute scores: dot product with channel sensitivities
    factor_cols = [c for c in channels.columns if c.startswith("f_")]
    ch = channels.copy()
    # Fill NaNs with 0 for safety
    ch[factor_cols] = ch[factor_cols].fillna(0.0)

    # Align columns
    missing = set(uw_df.columns) - set(factor_cols)
    if missing:
        st.error(f"The following factors are missing in channels.csv: {sorted(missing)}")
    else:
        scores = np.dot(ch[factor_cols].values, uw_df.values.T).reshape(-1)
        ch["score"] = scores

        top3 = ch.sort_values("score", ascending=False).head(3)

        st.subheader("Top 3 Matches")
        for _, row in top3.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['channel_name']}")
                st.markdown(f"**Score:** {row['score']:.2f}")
                tags = (row.get("tags") or "").strip()
                if tags:
                    st.markdown(f"**Tags:** {tags}")
                why = (row.get("why_fit_short") or "").strip()
                if why:
                    st.markdown(f"_{why}_")
                link = (row.get("compass_link") or "").strip()
                if link:
                    st.markdown(f"[Open Guidebook chapter]({link})")

        st.markdown("---")
        st.info("Want a brief ‘why these 3’? Enter your email below and click **Send me the why**.")
        email = st.text_input("Email address", key="email")
        send = st.button("Send me the why")
        if send:
            if "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                # Placeholder: In production, call your email service here.
                st.success("Got it! Your Top 3 summary will be emailed shortly.")
else:
    st.caption("Adjust the sliders, then click the button to see your matches.")
