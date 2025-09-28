# save_navigation_planner.py
from pathlib import Path
from weasyprint import HTML

def save_navigation_planner(user_id: str, html_content: str, drive_folder: str):
    """
    Generate a PDF from HTML and save into a Google Drive–synced folder.

    Args:
        user_id (str): Unique ID for the user.
        html_content (str): HTML content for the PDF.
        drive_folder (str): Path to local Google Drive folder (synced).
    """
    # Ensure folder exists
    folder = Path(drive_folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Unique filename for this user
    filename = folder / f"navigation_planner_{user_id}.pdf"

    # Render PDF
    HTML(string=html_content).write_pdf(str(filename))

    print(f"✅ Saved PDF for user {user_id} at {filename}")
    return filename

# Example test
if __name__ == "__main__":
    user_id = "test123"
    html_content = """
    <html>
    <body>
        <h1>Navigation Planner for Test User</h1>
        <p>This is a placeholder PDF for user_id = test123.</p>
    </body>
    </html>
    """
    # Change this path to your Google Drive synced folder
    drive_folder = "/Users/yourname/Google Drive/Compass_PDFs"
    save_navigation_planner(user_id, html_content, drive_folder)
