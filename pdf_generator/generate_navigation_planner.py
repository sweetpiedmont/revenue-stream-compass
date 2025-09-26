# generate_navigation_planner.py

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

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

def generate_pdf(user_name, top5, channels, outpath="planner.pdf"):
    html_content = render_navigation_html(user_name, top5, channels)
    HTML(string=html_content).write_pdf(outpath)
    print(f"✅ PDF written to {outpath}")

if __name__ == "__main__":
    # Fake test data
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

    generate_pdf(user_name, top5, channels)
