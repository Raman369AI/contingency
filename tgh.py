from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Fleet Graphics Quote Request", ln=True, align='C')

pdf.ln(10)
pdf.set_font("Arial", size=10)
pdf.cell(0, 10, "Customer Information", ln=True)
pdf.cell(0, 8, "Name: John Doe", ln=True)
pdf.cell(0, 8, "Email: johndoe@example.com", ln=True)
pdf.cell(0, 8, "Phone: 555-123-4567", ln=True)

pdf.ln(5)
pdf.cell(0, 10, "Graphics Request", ln=True)
pdf.cell(0, 8, "Type of Graphic: Vehicle Wrap", ln=True)
pdf.cell(0, 8, "Dimensions: 200 in x 60 in", ln=True)
pdf.cell(0, 8, "Quantity: 3", ln=True)
pdf.cell(0, 8, "Color Scheme: Blue and White", ln=True)
pdf.cell(0, 8, "Additional Notes: Please include company logo on both sides.", ln=True)

pdf.ln(10)
pdf.cell(0, 8, "Thank you for your request!", ln=True)

pdf.output("dummy_fleet_graphics_quote.pdf")
