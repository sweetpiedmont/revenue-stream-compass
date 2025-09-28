# generate_navigation_planner.py

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from save_to_drive import save_navigation_planner   # 👈 NEW IMPORT

# Set up Jinja environment (points to templates folder)
env = Environment(loader=FileSystemLoader("templates"))

def render_navigation_html(user_name, top5, channels):
    """Render HTML from Jinja template with given data."""
    template = env.get_template("planner.html")
    return template.render(
        user_name=user_name,
        top5=top5,
        channels=channels,
    )

def generate_pdf(user_id, user_name, top5, channels, outpath="planner.pdf"):
    """
    Generate the Navigation Planner PDF and save both locally and to Drive.
    """
    html_content = render_navigation_html(user_name, top5, channels)

    # Write locally
    HTML(string=html_content).write_pdf(outpath)
    print(f"✅ PDF written to {outpath}")

    # ALSO write to Drive (using the helper in save_to_drive.py)
    drive_folder = "/Users/sharon/Library/CloudStorage/GoogleDrive-hello@sweetpiedmontacademy.com/My Drive/Navigation Planner Storage"  # 👈 update this path
    save_navigation_planner(user_id, html_content, drive_folder)

if __name__ == "__main__":
    # Fake test data
    user_id = "test123"
    user_name = "Test User"
    top5 = ["Workshops", "Full Service Weddings", "Farmers Markets", "Subscriptions", "DIY Buckets"]

    # Fake channel data (simulate 2 channels for now)
    channels = [
        {
            "name": "Workshops",
            "score": 0.95,
            "narrative": "Workshops are a great way to engage directly with your community and generate steady income.",
            "advantages": ["Direct customer interaction", "High margins", "Community building"],
            "obstacles": ["Requires teaching skills", "Marketing is critical", "Seasonal interest may vary"],
        },
        {
            "name": "Full Service Weddings",
            "score": 0.92,
            "narrative": "Full service weddings provide high revenue but demand detailed planning and reliable staffing.",
            "advantages": ["High profit potential", "Strong demand", "Creative freedom"],
            "obstacles": ["High stress events", "Lots of moving parts", "Requires design expertise"],
        },
    ]

    generate_pdf(user_id, user_name, top5, channels)
