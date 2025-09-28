# generate_mini_report.py

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from save_to_drive import save_navigation_planner  # we’ll reuse this helper for Drive saves
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rsc_from_excel import load_from_excel, build_results

# --- Jinja2 setup ---
env = Environment(loader=FileSystemLoader("templates"))

def render_mini_report_html(user_name, top5, all_streams):
    """Render Mini-Report HTML from Jinja template with given data."""
    template = env.get_template("mini_report.html")
    return template.render(
        user_name=user_name,
        top5=top5,
        all_streams=all_streams,
    )

def generate_mini_report(user_id, user_name, top5, all_streams, outpath="mini_report.pdf"):
    """Generate Mini-Report PDF and save locally + to Google Drive."""
    html_content = render_mini_report_html(user_name, top5, all_streams)

    # Local write
    HTML(string=html_content).write_pdf(outpath)
    print(f"✅ Mini-Report written to {outpath}")

    # Google Drive write
    drive_folder = "/Users/sharon/Library/CloudStorage/GoogleDrive-hello@sweetpiedmontacademy.com/My Drive/Mini Report Storage"

    folder = Path(drive_folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"mini_report_{user_id}.pdf"
    outpath_drive = folder / filename

    HTML(string=html_content).write_pdf(str(outpath_drive))
    print(f"✅ Saved Mini-Report for user {user_id} at {outpath_drive}")


if __name__ == "__main__":
    # Accept user_id as argument
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = "test123"

    # --- Pull test data from Airtable ---
    user_name = "Test User"

    factors, categories, channels, narratives = load_from_excel(
        Path("Extreme_Weighting_Scoring_Prototype_for_FormWise_REPAIRED.xlsx")
    )

    # TEMP: pretend every slider was set to 5
    user_scores = {row["factor_id"]: 5 for _, row in factors.iterrows()}

    # Use the results builder (only needs user_scores, narratives, channels)
    top5, all_streams = build_results(user_scores, narratives, channels)

    generate_mini_report(user_id, user_name, top5, all_streams)

