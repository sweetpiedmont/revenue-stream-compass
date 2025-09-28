# generate_mini_report.py

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from save_to_drive import save_navigation_planner  # we’ll reuse this helper for Drive saves
import sys

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

    # --- Fake test data (replace later with Airtable pull) ---
    user_name = "Test User"

    top5 = [
        {"name": "Workshops", "score": 0.95, "narrative": "Workshops connect you directly with your community."},
        {"name": "Full Service Weddings", "score": 0.92, "narrative": "High revenue potential but requires careful planning."},
        {"name": "Farmers Markets", "score": 0.87, "narrative": "A steady, visible way to sell and connect locally."},
        {"name": "Subscriptions", "score": 0.83, "narrative": "Reliable recurring revenue if you have steady production."},
        {"name": "DIY Buckets", "score": 0.81, "narrative": "Simple, accessible way to serve casual events or budget-conscious customers."},
    ]

    all_streams = [
        {"name": "Workshops", "score": 0.95},
        {"name": "Full Service Weddings", "score": 0.92},
        {"name": "Farmers Markets", "score": 0.87},
        {"name": "Subscriptions", "score": 0.83},
        {"name": "DIY Buckets", "score": 0.81},
        {"name": "Wholesale to Florists", "score": 0.79},
        {"name": "Community & Corporate Events", "score": 0.76},
        {"name": "Pick-Your-Own Events", "score": 0.73},
        {"name": "Photography Venue", "score": 0.70},
        {"name": "Hotel & Restaurant Contracts", "score": 0.68},
        {"name": "Growers Collective", "score": 0.66},
        {"name": "Stockist / Consignment", "score": 0.63},
        {"name": "Wholesale Mixed Bouquets", "score": 0.61},
        {"name": "Farm Stand / Shop", "score": 0.60},
        {"name": "Subscriptions Lite", "score": 0.58},
        {"name": "DIY Bulk Sales", "score": 0.55},
        {"name": "Workshops (Advanced)", "score": 0.53},
        {"name": "CSA Hybrid", "score": 0.50},
    ]

    generate_mini_report(user_id, user_name, top5, all_streams)
