import streamlit as st
import pytesseract
import re
import pandas as pd
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
from datetime import datetime
import json
import io
from transformers import pipeline
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Application configuration"""
    EMISSION_FACTORS = {
        'electricity': 0.4,  # kg CO2e per kWh
        'natural_gas': 5.3,  # kg CO2e per therm
        'gasoline': 8.89,    # kg CO2e per gallon
        'diesel': 10.21,     # kg CO2e per gallon
        'coal': 2.23,        # kg CO2e per pound
        'propane': 5.75,     # kg CO2e per gallon
        'heating_oil': 10.15, # kg CO2e per gallon
        'jet_fuel': 9.57     # kg CO2e per gallon
    }
    
    UNIT_PATTERNS = {
        'kwh': 'electricity',
        'kilowatt': 'electricity',
        'therm': 'natural_gas',
        'mcf': 'natural_gas',
        'ccf': 'natural_gas',
        'gallon': 'gasoline',
        'gal': 'gasoline',
        'liter': 'gasoline',
        'btu': 'natural_gas'
    }

# --- Enhanced NLP Pipeline ---
@st.cache_resource
def load_models():
    """Load NLP models with caching"""
    try:
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        return ner_pipeline
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

# --- Gemini Integration ---
class GeminiAnalyst:
    """Integrates Google Gemini for intelligent data analysis"""
    
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.model = None
    
    def analyze_emissions_data(self, results: List[Dict]) -> str:
        """Analyze emissions data using Gemini with a structured prompt."""
        if not self.model:
            return "**AI Analyst Disabled:** Gemini API key not configured."

        try:
            total_emissions = sum(r.get('total_emissions', 0) for r in results)
            energy_breakdown = {}
            for result in results:
                for energy_type, data in result.get('extracted_data', {}).items():
                    if energy_type not in energy_breakdown:
                        energy_breakdown[energy_type] = {'amount': 0, 'emissions': 0}
                    energy_breakdown[energy_type]['amount'] += data.get('amount', 0)
                    energy_breakdown[energy_type]['emissions'] += data.get('emissions_kg_co2e', 0)

            prompt = f"""
            You are an expert emissions analyst. Generate a professional, structured report based on the following data.
            Use Markdown for formatting. Do not include any conversational introductory or concluding phrases.

            **Input Data:**
            - Total Emissions: {total_emissions:.2f} kg CO2e
            - Number of documents processed: {len(results)}
            - Energy Breakdown (kg CO2e):
              {json.dumps(energy_breakdown, indent=2)}

            **Report Structure:**

            ### 1. Key Insights
            - (Provide 3-5 bullet points highlighting the most important findings from the data.)

            ### 2. Emission Reduction Recommendations
            - (Provide a bulleted list of actionable recommendations based on the emissions breakdown.)

            ### 3. Areas of Concern
            - (Provide a bulleted list of potential issues or areas that require further investigation.)

            ### 4. Benchmarking
            - (Provide a brief comparison to typical emissions for a similar entity, if possible. State if not enough data is available for a direct comparison.)
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return f"**Analysis Failed:** An error occurred while generating the AI analysis: {str(e)}"
    
    def chat_about_data(self, user_question: str, results: List[Dict]) -> str:
        """Handle user questions about the data"""
        if not self.model:
            return "Gemini API key not configured. Please add your API key to enable chat functionality."
        
        try:
            context = f"""
            You are an expert emissions analyst. Here's the current data context:
            
            Total documents: {len(results)}
            Total emissions: {sum(r.get('total_emissions', 0) for r in results):.2f} kg CO2e
            
            Data summary: {json.dumps([{
                'file': r['filename'],
                'emissions': r['total_emissions'],
                'energy_types': list(r.get('extracted_data', {}).keys())
            } for r in results], indent=2)}
            
            User question: {user_question}
            
            Please provide a helpful, accurate response based on the data.
            """
            
            response = self.model.generate_content(context)
            return response.text
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Sorry, I couldn't process your question: {str(e)}"

# --- Enhanced Extractor ---
class AdvancedCarbonExtractor:
    """Enhanced extraction with better accuracy and error handling"""
    
    def __init__(self, ner_pipeline):
        self.ner_pipeline = ner_pipeline
        self.config = Config()
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Enhanced OCR with preprocessing"""
        try:
            # Preprocess image for better OCR
            image = image.convert('RGB')
            
            # Apply OCR with custom config
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""
    
    def extract_energy_data(self, text: str, filename: str) -> Dict[str, Any]:
        """Enhanced extraction with better pattern matching"""
        results = {
            'filename': filename,
            'raw_text': text,
            'extracted_data': {},
            'total_emissions': 0,
            'extraction_date': datetime.now().isoformat(),
            'inconsistencies': [],
            'confidence_score': 0
        }
        
        if not text.strip():
            results['inconsistencies'].append("No text could be extracted from the image.")
            return results
        
  
        extracted_items = self._extract_with_patterns(text)
       
        if self.ner_pipeline:
            ner_items = self._extract_with_ner(text)
            extracted_items.extend(ner_items)
        
      
        processed_data = self._process_extracted_items(extracted_items)
        
        results['extracted_data'] = processed_data
        results['total_emissions'] = sum(
            data.get('emissions_kg_co2e', 0) 
            for data in processed_data.values()
        )
        
        # Calculate confidence score
        results['confidence_score'] = self._calculate_confidence(processed_data, text)
        
        if not processed_data:
            results['inconsistencies'].append(
                "No energy consumption data could be reliably extracted."
            )
        
        return results
    
    def _extract_with_patterns(self, text: str) -> List[Dict]:
        """Extract using regex patterns"""
        items = []
  
        patterns = [
            # Electricity: 1,234 kWh, 1234.56 kwh
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(kwh|kilowatt[- ]?hours?)', 'electricity'),
            # Natural gas: 123 therms, 45.6 mcf
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(therms?|mcf|ccf)', 'natural_gas'),
            # Fuel: 45.2 gallons, 123.4 gal
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(gallons?|gal)', 'gasoline'),
            # General BTU pattern
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(btu)', 'natural_gas'),
        ]
        
        for pattern, energy_type in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    unit = match.group(2)
                    
                    items.append({
                        'amount': amount,
                        'unit': unit,
                        'energy_type': energy_type,
                        'source_text': match.group(0),
                        'method': 'regex'
                    })
                except ValueError:
                    continue
        
        return items
    
    def _extract_with_ner(self, text: str) -> List[Dict]:
        """Extract using NER model"""
        items = []
        
        try:
            entities = self.ner_pipeline(text)
            
            for entity in entities:
                if entity['entity_group'] in ['QUANTITY', 'CARDINAL']:
                    # Look for energy-related terms in the entity
                    entity_text = entity['word'].lower()
                    
                    for unit_pattern, energy_type in self.config.UNIT_PATTERNS.items():
                        if unit_pattern in entity_text:
                            # Extract number
                            numbers = re.findall(r'\d+\.?\d*', entity['word'])
                            if numbers:
                                try:
                                    amount = float(numbers[0])
                                    items.append({
                                        'amount': amount,
                                        'unit': unit_pattern,
                                        'energy_type': energy_type,
                                        'source_text': entity['word'],
                                        'method': 'ner',
                                        'confidence': entity.get('score', 0)
                                    })
                                except ValueError:
                                    continue
        except Exception as e:
            logger.error(f"NER extraction error: {e}")
        
        return items
    
    def _process_extracted_items(self, items: List[Dict]) -> Dict[str, Dict]:
        """Process and deduplicate extracted items"""
        processed = {}
        
        for item in items:
            energy_type = item['energy_type']
            amount = item['amount']
         
            emission_factor = self.config.EMISSION_FACTORS.get(energy_type, 0)
            emissions = amount * emission_factor
            
            if energy_type in processed:
                existing = processed[energy_type]
                if amount > existing['amount'] or item.get('confidence', 0) > existing.get('confidence', 0):
                    processed[energy_type] = {
                        'amount': amount,
                        'unit': item['unit'],
                        'emissions_kg_co2e': emissions,
                        'source_text': item['source_text'],
                        'method': item['method'],
                        'confidence': item.get('confidence', 0)
                    }
            else:
                processed[energy_type] = {
                    'amount': amount,
                    'unit': item['unit'],
                    'emissions_kg_co2e': emissions,
                    'source_text': item['source_text'],
                    'method': item['method'],
                    'confidence': item.get('confidence', 0)
                }
        
        return processed
    
    def _calculate_confidence(self, data: Dict, text: str) -> float:
        """Calculate overall confidence score"""
        if not data:
            return 0.0
        
        scores = []
        for item in data.values():
            # Base confidence from extraction method
            base_score = 0.8 if item['method'] == 'regex' else item.get('confidence', 0.6)
            
            # Boost if we found clear unit indicators
            if any(unit in text.lower() for unit in ['kwh', 'therm', 'gallon']):
                base_score += 0.1
            
            # Boost if amounts seem reasonable
            if 0 < item['amount'] < 10000:
                base_score += 0.1
            
            scores.append(min(base_score, 1.0))
        
        return sum(scores) / len(scores) if scores else 0.0

# --- Enhanced Visualization ---
def create_emissions_charts(results: List[Dict]):
    """Create interactive charts for emissions data"""
    
    # Prepare data
    chart_data = []
    for result in results:
        for energy_type, data in result.get('extracted_data', {}).items():
            chart_data.append({
                'File': result['filename'],
                'Energy Type': energy_type.replace('_', ' ').title(),
                'Amount': data.get('amount', 0),
                'Emissions': data.get('emissions_kg_co2e', 0),
                'Unit': data.get('unit', '')
            })
    
    if not chart_data:
        return None, None
    
    df = pd.DataFrame(chart_data)
    
    # Emissions by energy type
    fig1 = px.pie(df, values='Emissions', names='Energy Type', 
                  title='Emissions by Energy Type (kg CO2e)')
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    
    # Emissions by file
    fig2 = px.bar(df, x='File', y='Emissions', color='Energy Type',
                  title='Emissions by Document')
    fig2.update_layout(xaxis_tickangle=-45)
    
    return fig1, fig2

# --- Enhanced Report Generation ---
def generate_professional_pdf_report(results: List[Dict], analysis: str = None) -> io.BytesIO:
    """Generate a professional PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86C1'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1B4F72'),
        spaceAfter=12
    )
    
    # Title page
    story.append(Paragraph("Emissions Audit Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    total_emissions = sum(r.get('total_emissions', 0) for r in results)
    total_files = len(results)
    
    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Documents Processed', str(total_files)],
        ['Total Emissions', f'{total_emissions:.2f} kg CO2e'],
        ['Average per Document', f'{total_emissions/total_files:.2f} kg CO2e' if total_files > 0 else 'N/A'],
        ['Report Date', datetime.now().strftime('%Y-%m-%d %H:%M')]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    # Detailed Results
    story.append(Paragraph("Detailed Results", heading_style))
    
    for result in results:
        story.append(Paragraph(f"Document: {result['filename']}", styles['Heading3']))
        
        if result.get('extracted_data'):
            data_rows = [['Energy Type', 'Consumption', 'Unit', 'Emissions (kg CO2e)']]
            
            for energy_type, data in result['extracted_data'].items():
                data_rows.append([
                    energy_type.replace('_', ' ').title(),
                    f"{data.get('amount', 'N/A'):.2f}",
                    data.get('unit', ''),
                    f"{data.get('emissions_kg_co2e', 0):.2f}"
                ])
            
            detail_table = Table(data_rows)
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#85C1E9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF5FB')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(detail_table)
        else:
            story.append(Paragraph("No energy data extracted from this document.", styles['Normal']))
        
        # Confidence and issues
        confidence = result.get('confidence_score', 0)
        story.append(Paragraph(f"Confidence Score: {confidence:.1%}", styles['Normal']))
        
        if result.get('inconsistencies'):
            story.append(Paragraph("Issues Found:", styles['Heading4']))
            for issue in result['inconsistencies']:
                story.append(Paragraph(f"• {issue}", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    # AI Analysis (if available)
    if analysis and "API key not configured" not in analysis:
        story.append(PageBreak())
        story.append(Paragraph("AI Analysis & Recommendations", heading_style))
        story.append(Paragraph(analysis, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(
        page_title="Emissions Audit Pipeline",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #EBF5FB;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌱 Advanced Emissions Audit Pipeline</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload energy bills/documents",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=True,
            help="Upload images of energy bills, utility statements, or other consumption documents"
        )
    
    # Initialize session state
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = []
    if 'gemini_analyst' not in st.session_state:
        try:
            st.session_state.gemini_analyst = GeminiAnalyst()
        except ValueError as e:
            st.error(e)
            st.session_state.gemini_analyst = None
    
    # Process uploaded files
    if uploaded_files:
        ner_pipeline = load_models()
        if ner_pipeline is None:
            st.error("Failed to load NLP models. Please refresh the page.")
            return
        
        extractor = AdvancedCarbonExtractor(ner_pipeline)
        st.session_state.extraction_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    text = extractor.extract_text_from_image(image)
                    
                    if text.strip():
                        result = extractor.extract_energy_data(text, uploaded_file.name)
                        st.session_state.extraction_results.append(result)
                    else:
                        st.warning(f"No text could be extracted from {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        progress_bar.empty()
        status_text.empty()
    
    # Display results
    if st.session_state.extraction_results:
        results = st.session_state.extraction_results
        
        # Metrics overview
        st.header("📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_emissions = sum(r.get('total_emissions', 0) for r in results)
        avg_confidence = sum(r.get('confidence_score', 0) for r in results) / len(results)
        total_files = len(results)
        successful_extractions = len([r for r in results if r.get('extracted_data')])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌍 Total Emissions</h3>
                <h2>{total_emissions:.2f} kg CO2e</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄 Documents</h3>
                <h2>{successful_extractions}/{total_files}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Avg Confidence</h3>
                <h2>{avg_confidence:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            co2_equivalent = total_emissions * 2.2  # rough trees planted equivalent
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌳 Trees to Offset</h3>
                <h2>{co2_equivalent:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.header("📈 Data Visualization")
        fig1, fig2 = create_emissions_charts(results)
        
        if fig1 and fig2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed data table
        st.header("📋 Detailed Results")
        display_data = []
        for result in results:
            for energy_type, data in result.get('extracted_data', {}).items():
                display_data.append({
                    "Document": result['filename'],
                    "Energy Type": energy_type.replace('_', ' ').title(),
                    "Consumption": f"{data.get('amount', 'N/A'):.2f}",
                    "Unit": data.get('unit', ''),
                    "Emissions (kg CO2e)": f"{data.get('emissions_kg_co2e', 0):.2f}",
                    "Confidence": f"{result.get('confidence_score', 0):.1%}",
                    "Method": data.get('method', 'N/A').title()
                })
        
        if display_data:
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)
        
        # AI Analysis
        if st.session_state.gemini_analyst:
            st.header("🤖 AI Analysis & Insights")
            
            if st.button("Generate AI Analysis", type="primary"):
                with st.spinner("Analyzing data with AI..."):
                    analysis = st.session_state.gemini_analyst.analyze_emissions_data(results)
                    st.session_state.ai_analysis = analysis
            
            if hasattr(st.session_state, 'ai_analysis'):
                st.markdown(st.session_state.ai_analysis)
            
            # Chat interface
            st.subheader("💬 Chat with Your Data")
            user_question = st.text_input("Ask questions about your emissions data:")
            
            if user_question:
                with st.spinner("Getting answer..."):
                    answer = st.session_state.gemini_analyst.chat_about_data(user_question, results)
                    st.markdown(f"**Answer:** {answer}")
        
        # Data quality issues
        st.header("⚠️ Data Quality & Issues")
        issues_found = False
        for result in results:
            if result.get('inconsistencies'):
                issues_found = True
                with st.expander(f"Issues in {result['filename']} (Confidence: {result.get('confidence_score', 0):.1%})"):
                    for issue in result['inconsistencies']:
                        st.warning(issue)
        
        if not issues_found:
            st.success("No data quality issues detected!")
        
        # Download reports
        st.header("📥 Download Reports")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Excel report
            excel_data = generate_excel_report(results)
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"emissions_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Enhanced PDF report
            analysis_text = getattr(st.session_state, 'ai_analysis', None)
            pdf_data = generate_professional_pdf_report(results, analysis_text)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name=f"emissions_audit_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        
        with col3:
            # JSON data export
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="💾 Download Raw Data (JSON)",
                data=json_data,
                file_name=f"emissions_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        st.info("📤 Upload energy bills or consumption documents to begin the automated emissions accounting process.")
        st.markdown("""
                    
        If you not having any documents to upload, you can still explore the features of this application. download the sample documents from the [Drive](https://drive.google.com/drive/folders/1thkld5yx_1ghEgVmm4cAhBhvvABnD2Jn?usp=sharing) and upload them here.
                    
        ### How it works:
        1. **Upload** images of energy bills, utility statements, or consumption documents
        2. **Extract** data using advanced OCR and AI models
        3. **Calculate** emissions using standard factors
        4. **Analyze** results with AI-powered insights
        5. **Download** professional reports
        
        ### Supported formats:
        - Images: PNG, JPG, JPEG
        - Energy types: Electricity, Natural Gas, Gasoline, Diesel, and more
        """)

def generate_excel_report(results):
    """Generate enhanced Excel report"""
    display_data = []
    summary_data = []
    
    for result in results:
        # Detailed data
        for energy_type, data in result.get('extracted_data', {}).items():
            display_data.append({
                "Document": result['filename'],
                "Energy Type": energy_type.replace('_', ' ').title(),
                "Amount": data.get('amount', 0),
                "Unit": data.get('unit', ''),
                "Emissions (kg CO2e)": data.get('emissions_kg_co2e', 0),
                "Confidence": result.get('confidence_score', 0),
                "Extraction Method": data.get('method', 'N/A').title(),
                "Source Text": data.get('source_text', ''),
                "Date Processed": result.get('extraction_date', '')
            })
        
        # Summary data
        summary_data.append({
            "Document": result['filename'],
            "Total Emissions (kg CO2e)": result.get('total_emissions', 0),
            "Confidence Score": result.get('confidence_score', 0),
            "Energy Types Found": len(result.get('extracted_data', {})),
            "Issues Count": len(result.get('inconsistencies', [])),
            "Processing Date": result.get('extraction_date', '')
        })
    
    # Create Excel file with multiple sheets
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Detailed sheet
        if display_data:
            df_detailed = pd.DataFrame(display_data)
            df_detailed.to_excel(writer, index=False, sheet_name='Detailed Results')
            
            # Format the detailed sheet
            workbook = writer.book
            worksheet = writer.sheets['Detailed Results']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#2E86C1',
                'font_color': 'white',
                'border': 1
            })
            
            # Write headers with formatting
            for col_num, value in enumerate(df_detailed.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(df_detailed.columns):
                max_length = max(df_detailed[col].astype(str).map(len).max(), len(col))
                worksheet.set_column(i, i, min(max_length + 2, 50))
        
        # Summary sheet
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            # Add totals row
            totals_row = {
                "Document": "TOTAL",
                "Total Emissions (kg CO2e)": df_summary["Total Emissions (kg CO2e)"].sum(),
                "Confidence Score": df_summary["Confidence Score"].mean(),
                "Energy Types Found": df_summary["Energy Types Found"].sum(),
                "Issues Count": df_summary["Issues Count"].sum(),
                "Processing Date": datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            
            # Append totals
            df_with_totals = pd.concat([df_summary, pd.DataFrame([totals_row])], ignore_index=True)
            
            df_with_totals.to_excel(writer, index=False, sheet_name='Summary', startrow=0)
        
        # Issues sheet
        issues_data = []
        for result in results:
            for issue in result.get('inconsistencies', []):
                issues_data.append({
                    "Document": result['filename'],
                    "Issue": issue,
                    "Confidence": result.get('confidence_score', 0),
                    "Date": result.get('extraction_date', '')
                })
        
        if issues_data:
            df_issues = pd.DataFrame(issues_data)
            df_issues.to_excel(writer, index=False, sheet_name='Issues & Warnings')
    
    output.seek(0)
    return output.getvalue()


if __name__ == "__main__":
    main()