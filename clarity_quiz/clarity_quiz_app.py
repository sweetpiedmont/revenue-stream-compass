
import streamlit as st
from collections import Counter

st.set_page_config(page_title="Clarity Pack Quiz", page_icon="🌸", layout="centered")

st.title("✨ Where Are You On Your Flower Farming Journey?")
st.subheader("Take this 2-minute check-in to discover your flower farm stage — and get your free Clarity Pack to help you Live the Flower Life YOUR way.")

# --- Questions ---
questions = {
    "Q1: Have you sold flowers for money yet?": [
        ("🌱 No, but I dream of it.", "Seedling"),
        ("🌸 Yes, just a few times or as a tiny side hustle.", "Budding"),
        ("💐 Yes, regularly — it’s part of my income.", "Blooming")
    ],
    "Q2: How many seasons have you grown flowers with selling in mind?": [
        ("🌱 0 — I’m just researching or planting for fun.", "Seedling"),
        ("🌸 1–2 seasons.", "Budding"),
        ("💐 3 or more seasons.", "Blooming")
    ],
    "Q3: Which feels most like you right now?": [
        ("🌱 I’m dreaming and gathering ideas — not sure where to start.", "Seedling"),
        ("🌸 I’ve dipped my toes in — but I want more focus and less overwhelm.", "Budding"),
        ("💐 I’ve built a flower business — but I want it to fit my life better.", "Blooming")
    ],
    "Q4: How do you feel about your flower farm dream today?": [
        ("🌱 I want to plan wisely before I spend money or time.", "Seedling"),
        ("🌸 I want to grow this dream bigger — but keep my sanity.", "Budding"),
        ("💐 I want to refine and protect what I’ve built, so it stays joyful.", "Blooming")
    ],
    "Q5 (Optional): How much time do you spend on flowers now?": [
        ("🌱 Just for fun, in my spare time.", "Seedling"),
        ("🌸 A few hours a week — juggling it with other work.", "Budding"),
        ("💐 It takes up a big chunk of my week.", "Blooming")
    ],
}

responses = {}
for q, options in questions.items():
    responses[q] = st.radio(q, [opt[0] for opt in options], index=None)

if st.button("👉 See My Result"):
    # Count stage choices
    stages = []
    for q, choice in responses.items():
        if choice:
            for opt, stage in questions[q]:
                if opt == choice:
                    stages.append(stage)
    if not stages:
        st.warning("Please answer at least one question.")
    else:
        # Majority stage
        stage = Counter(stages).most_common(1)[0][0]

        if stage == "Seedling":
            st.success("🌱 You’re in the Seedling Stage — and that’s a beautiful place to begin!")
            st.write("You’re dreaming wisely before you leap. Good for you!\n\n"
                     "Your free **Seedling Clarity Pack** helps you name how you want your flower life to feel, "
                     "what you want (and don’t want), and what matters most — so you can plan your next steps with confidence.\n\n"
                     "👉 [Download Your Seedling Clarity Pack](#)")
        elif stage == "Budding":
            st.success("🌸 You’re in the Budding Stage — it’s time to grow smarter, not harder!")
            st.write("You’ve planted seeds and sold your first stems — now you want to grow wisely, protect your energy, "
                     "and keep Living the Flower Life your way.\n\n"
                     "Your free **Budding Clarity Pack** helps you pause, reflect, and see what you want more of and what you don’t — "
                     "so you can plan your next season with clarity.\n\n"
                     "👉 [Download Your Budding Clarity Pack](#)")
        elif stage == "Blooming":
            st.success("💐 You’re in the Blooming Stage — your next season deserves focus and joy!")
            st.write("You’ve built a flower business you can be proud of — but you know staying in bloom takes intention.\n\n"
                     "Your free **Blooming Clarity Pack** helps you remember what you love, name what drains you, and choose what stays and what goes — "
                     "so your flower farm fits your life, not the other way around.\n\n"
                     "👉 [Download Your Blooming Clarity Pack](#)")
