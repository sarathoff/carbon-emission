
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import pytesseract
import re
from datetime import datetime
import pandas as pd
from transformers import pipeline
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)

# --- NLP Model and Configuration ---
def load_ner_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def load_summarization_pipeline():
    return pipeline("summarization", model="google/flan-t5-base")

ner_pipeline = load_ner_pipeline()
summarization_pipeline = load_summarization_pipeline()

class CarbonExtractor:
    def __init__(self):
        self.emission_factors = {
            'electricity': 0.4, 'natural_gas': 5.3, 'gasoline': 8.89,
            'diesel': 10.21, 'coal': 2.23, 'propane': 5.75
        }

    def extract_text_from_image(self, image):
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            return str(e)

    def extract_energy_data(self, text, filename):
        results = {
            'filename': filename, 'raw_text': text, 'extracted_data': {},
            'total_emissions': 0, 'extraction_date': datetime.now().isoformat(),
            'inconsistencies': []
        }
        ner_results = ner_pipeline(text)

        if not ner_results:
            results['inconsistencies'].append("No entities found in text.")

        for entity in ner_results:
            energy_type = None
            if 'QUANTITY' in entity['entity_group'] or 'CARDINAL' in entity['entity_group']:
                if 'kwh' in entity['word'].lower():
                    energy_type = 'electricity'
                elif 'therm' in entity['word'].lower():
                    energy_type = 'natural_gas'
                elif 'gallon' in entity['word'].lower():
                    energy_type = 'gasoline'

            if energy_type:
                try:
                    amount = float(re.findall(r'\d+\.?\d*', entity['word'])[0])
                    emissions = amount * self.emission_factors.get(energy_type, 0)
                    results['extracted_data'][energy_type] = {
                        'amount': amount, 'unit': entity['word'],
                        'emissions_kg_co2e': emissions
                    }
                    results['total_emissions'] += emissions
                except (ValueError, IndexError):
                    results['inconsistencies'].append(f"Could not parse amount from: {entity['word']}")
        
        if not results['extracted_data']:
            results['inconsistencies'].append("No relevant energy data could be extracted from the document.")
            
        return results

def generate_summary(text):
    summary = summarization_pipeline(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_pdf_report(results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Emissions Audit Report", styles['h1']))
    elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", styles['h3']))
    elements.append(Spacer(1, 0.25*inch))

    for result in results:
        elements.append(Paragraph(f"Document: {result['filename']}", styles['h2']))
        
        data = [["Energy Type", "Consumption", "Emissions (kg CO2e)"]]
        for energy_type, d in result.get('extracted_data', {}).items():
            data.append([
                energy_type.replace('_', ' ').title(),
                f"{d.get('amount', 'N/A')} {d.get('unit', '')}",
                f"{d.get('emissions_kg_co2e', 0):.2f}"
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.25*inch))

        if result.get('inconsistencies'):
            elements.append(Paragraph("Inconsistencies Found:", styles['h3']))
            for issue in result['inconsistencies']:
                elements.append(Paragraph(f"- {issue}", styles['Normal']))
        elements.append(Spacer(1, 0.5*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    results = []
    extractor = CarbonExtractor()

    for file in files:
        image = Image.open(file)
        text = extractor.extract_text_from_image(image)
        if "Error" not in text:
            result = extractor.extract_energy_data(text, file.filename)
            results.append(result)

    return jsonify(results)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    summary = generate_summary(text)
    return jsonify({'summary': summary})

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    results = request.get_json()
    pdf_buffer = generate_pdf_report(results)
    return send_file(pdf_buffer, as_attachment=True, download_name='emissions_report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
