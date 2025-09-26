# generate_navigation_planner.py

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# Set up Jinja environment (points to templates folder)
env = Environment(loader=FileSystemLoader("templates"))

def render_navigation_html(user_name, top5, rackstack):
    """Render HTML from Jinja template with given data."""
    template = env.get_template("planner.html")
    return template.render(
        user_name=user_name,
        top5=top5,
        rackstack=rackstack,
    )

def generate_pdf(user_name, top5, rackstack, outpath="planner.pdf"):
    html_content = render_navigation_html(user_name, top5, rackstack)
    HTML(string=html_content).write_pdf(outpath)
    print(f"✅ PDF written to {outpath}")

if __name__ == "__main__":
    # Fake test data
    user_name = "Test User"
    top5 = ["Workshops", "Full Service Weddings", "Farmers Markets", "Subscriptions", "DIY Buckets"]
    rackstack = [
        ("Workshops", 0.95),
        ("Full Service Weddings", 0.92),
        ("Farmers Markets", 0.75),
        ("Subscriptions", 0.67),
        ("DIY Buckets", 0.60),
    ]
    generate_pdf(user_name, top5, rackstack)
