from fpdf import FPDF
from docx import Document

def generate_pdf(data_dict):
    """Generate a PDF report from the checklist."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(80, 10, text="Checks", border=1)
    pdf.cell(80, 10, text="Status", border=1)
    pdf.ln()
    
    pdf.set_font("Arial", size=12)
    for key, value in data_dict.items():
        pdf.cell(80, 10, text=str(key), border=1)
        pdf.cell(80, 10, text=str(value), border=1)
        pdf.ln()

    return pdf

def generate_docx(data_dict):
    """Generate a DOCX report from the checklist."""
    doc = Document()
    doc.add_heading('Inspection Report', 0)

    table = doc.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Checks'
    hdr_cells[1].text = 'Status'

    for key, value in data_dict.items():
        row_cells = table.add_row().cells
        row_cells[0].text = str(key)
        row_cells[1].text = str(value)

    return doc
