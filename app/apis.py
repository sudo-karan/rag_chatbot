def search_datasets(keyword: str) -> str:
    if not keyword.strip():
        return "Please provide a search term to find datasets."
    results = [
        {
            "title": f"Agricultural Data - {keyword.title()} Production 2023",
            "url": f"https://data.gov.in/dataset/{keyword.lower().replace(' ', '-')}-production-2023",
            "ministry": "Ministry of Agriculture",
            "last_updated": "2024-01-15",
        },
        {
            "title": f"District-wise {keyword.title()} Statistics",
            "url": f"https://data.gov.in/dataset/district-{keyword.lower().replace(' ', '-')}-stats",
            "ministry": "Ministry of Statistics",
            "last_updated": "2023-11-20",
        },
        {
            "title": f"{keyword.title()} Export Records - DGFT",
            "url": f"https://data.gov.in/dataset/{keyword.lower().replace(' ', '-')}-export-dgft",
            "ministry": "Ministry of Commerce",
            "last_updated": "2024-02-01",
        },
    ]
    lines = [f"Here are the top search results for '{keyword}':\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   Ministry: {r['ministry']}")
        lines.append(f"   Last updated: {r['last_updated']}")
        lines.append(f"   URL: {r['url']}\n")
    return "\n".join(lines)


def get_cdo_details(query: str) -> str:
    return (
        f"CDO details for query '{query}':\n\n"
        f"Dr. Priya Sharma\n"
        f"Designation: Chief Data Officer\n"
        f"Ministry: Ministry of Agriculture & Farmers Welfare\n"
        f"Email: cdo.agriculture@gov.in\n"
        f"Phone: +91-11-2338-XXXX\n"
        f"Office: Krishi Bhavan, New Delhi"
    )


def get_dataset_cdo(dataset_url: str) -> str:
    return (
        f"Dataset: {dataset_url}\n\n"
        f"Uploaded by: Dr. Priya Sharma (Chief Data Officer, Ministry of Agriculture)\n"
        f"Upload date: 2024-01-10\n"
        f"Contact: cdo.agriculture@gov.in\n"
        f"Last verified: 2024-03-01"
    )


def submit_portal_feedback(message: str = "") -> str:
    return (
        "Your feedback about the portal has been submitted successfully. "
        "Our team will review it within 3-5 working days. "
        "Thank you for helping us improve the Government Open Data Portal."
    )


def contact_cdo_or_dataset_feedback(dataset_ref: str = "") -> str:
    if dataset_ref:
        return (
            f"Your feedback/query regarding '{dataset_ref}' has been forwarded to the "
            f"responsible Chief Data Officer. They will respond within 5 working days. "
            f"For urgent matters, you may also email cdo.portal@gov.in directly."
        )
    return (
        "Your message has been forwarded to the relevant Chief Data Officer. "
        "You will receive a response within 5 working days. "
        "For urgent matters, contact cdo.portal@gov.in."
    )
