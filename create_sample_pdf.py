"""Run once to create the sample PDF: python create_sample_pdf.py"""
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", style="B", size=16)
pdf.cell(0, 10, "Guide to Oranges - Government Agricultural Portal", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)

content = [
    ("What is an Orange?",
     "An orange is a citrus fruit from the species Citrus sinensis. It is one of the most popular fruits "
     "in the world, known for its bright orange colour, sweet-tart flavour, and high vitamin C content. "
     "Oranges belong to the family Rutaceae and are cultivated in tropical and subtropical climates."),
    ("Nutritional Value",
     "A medium orange (approximately 131 grams) contains: 62 calories, 15.4 grams of carbohydrates, "
     "3.1 grams of fibre, 12.2 grams of sugar, 1.2 grams of protein, 70 mg of vitamin C (78% of daily value), "
     "and 14 mcg of folate. Oranges are also a good source of potassium, thiamine, and antioxidants."),
    ("Types of Oranges",
     "Common orange varieties include: Navel oranges (seedless, sweet, easy to peel), Valencia oranges "
     "(juicy, ideal for juicing), Blood oranges (red-purple flesh, berry-like flavour), Cara Cara oranges "
     "(pink flesh, low acidity), and Mandarin oranges (small, loose skin, very sweet)."),
    ("Orange Cultivation in India",
     "India is one of the top producers of oranges globally. Key producing states include Nagpur (Maharashtra), "
     "known as the Orange City, Coorg (Karnataka), and parts of Rajasthan and Punjab. The main harvesting "
     "season is November to March. The Nagpur orange has a Geographical Indication (GI) tag."),
    ("Health Benefits",
     "Oranges offer several health benefits: they boost immunity due to high vitamin C content, support heart "
     "health through flavonoids and potassium, aid digestion via dietary fibre, help prevent kidney stones, "
     "and contain antioxidants that reduce oxidative stress."),
    ("Storage and Handling",
     "Oranges can be stored at room temperature for up to one week or refrigerated for up to four weeks. "
     "They should be kept away from direct sunlight. Once peeled or cut, store in an airtight container "
     "and consume within two days."),
    ("Government Schemes for Orange Farmers",
     "The Ministry of Agriculture supports orange cultivation through the National Horticulture Mission (NHM), "
     "which provides subsidies for planting material, drip irrigation, and cold storage infrastructure. "
     "Farmers can register on this portal to access scheme information and connect with Chief Data Officers."),
]

for heading, body in content:
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, body)
    pdf.ln(3)

pdf.output("pdfs/sample_oranges.pdf")
print("Sample PDF created at pdfs/sample_oranges.pdf")
