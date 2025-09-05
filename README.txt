
Revenue Stream Compass™ — Streamlit MVP

Files:
- factors.csv — Field Factor catalog
- channels.csv — Revenue stream profiles with factor sensitivities
- rsc_streamlit_stub.py — Minimal Streamlit app that computes Top 3

Quick start:
1) Download all three files to the same folder.
2) In a terminal, run:
   pip install streamlit pandas numpy
   streamlit run rsc_streamlit_stub.py
3) The app will open in your browser. Move sliders, click "See my Top 3".

Next steps you can add:
- Real factor weights (add a weights UI and update the dot product).
- Presets (define dictionaries of default scores/weights).
- Email sending via Postmark/SendGrid API.
- Branding via .streamlit/config.toml.
- Multipage full Compass with rack-and-stack, compare views, and downloads.
