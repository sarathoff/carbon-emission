# unified_streamlit_app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile
import base64
from PIL import Image
import re

# Import our extraction classes
try:
    from cloud_ocr_extractor import CloudCarbonExtractor
except ImportError:
    # If import fails, create a simple version inline
    class CloudCarbonExtractor:
        def __init__(self, use_mock=True):
            self.emission_factors = {
                'electricity': 0.4,
                'natural_gas': 5.3,
                'gasoline': 8.89,
                'diesel': 10.21
            }
        
        def mock_ocr_extraction(self, image_path):
            filename = os.path.basename(image_path).lower()
            if 'electric' in filename:
                return "ELECTRIC UTILITY COMPANY\nTotal kWh Used: 1,230 kWh\nEnergy Charges: $147.60"
            elif 'gas' in filename:
                return "NATURAL GAS COMPANY\nUsage: 130 Therms\nGas Charges: $162.50"
            elif 'fuel' in filename:
                return "FUEL STATION RECEIPT\nGallons: 18.5\nRegular Gasoline\nTotal: $63.83"
            return "Sample energy document"
        
        def extract_text_from_image(self, image_path, api_key=None):
            return self.mock_ocr_extraction(image_path)
        
        def smart_extract_energy_data(self, text, filename):
            results = {
                'filename': filename,
                'extraction_date': datetime.now().isoformat(),
                'extracted_data': {},
                'total_emissions': 0,
                'document_type': 'unknown'
            }
            
            # Simple pattern matching
            patterns = {
                'electricity': r'(\d+\.?\d*)\s*kWh',
                'natural_gas': r'(\d+\.?\d*)\s*therms?',
                'gasoline': r'(\d+\.?\d*)\s*gallons?'
            }
            
            text_lower = text.lower()
            for energy_type, pattern in patterns.items():
                matches = re.findall(pattern, text_lower)
                if matches:
                    amount = float(matches[0])
                    emissions = amount * self.emission_factors.get(energy_type, 0)
                    results['extracted_data'][energy_type] = {
                        'amount': amount,
                        'unit': 'kWh' if energy_type == 'electricity' else 'therms' if energy_type == 'natural_gas' else 'gallons',
                        'emissions_kg_co2e': round(emissions, 2)
                    }
                    results['total_emissions'] += emissions
                    
                    if energy_type == 'electricity':
                        results['document_type'] = 'electricity_bill'
                    elif energy_type == 'natural_gas':
                        results['document_type'] = 'gas_bill'
                    elif energy_type == 'gasoline':
                        results['document_type'] = 'fuel_receipt'
            
            results['total_emissions'] = round(results['total_emissions'], 2)
            return results

def create_sample_data():
    """Create sample data for immediate demo"""
    sample_results = [
        {
            'filename': 'electric_bill_1.png',
            'extraction_date': datetime.now().isoformat(),
            'document_type': 'electricity_bill',
            'extracted_data': {
                'electricity': {
                    'amount': 1230,
                    'unit': 'kWh',
                    'emissions_kg_co2e': 492.0
                }
            },
            'total_emissions': 492.0
        },
        {
            'filename': 'gas_bill_1.png',
            'extraction_date': datetime.now().isoformat(),
            'document_type': 'gas_bill',
            'extracted_data': {
                'natural_gas': {
                    'amount': 130,
                    'unit': 'therms',
                    'emissions_kg_co2e': 689.0
                }
            },
            'total_emissions': 689.0
        },
        {
            'filename': 'fuel_receipt_1.png',
            'extraction_date': datetime.now().isoformat(),
            'document_type': 'fuel_receipt',
            'extracted_data': {
                'gasoline': {
                    'amount': 18.5,
                    'unit': 'gallons',
                    'emissions_kg_co2e': 164.47
                }
            },
            'total_emissions': 164.47
        }
    ]
    
    # Generate summary
    total_emissions = sum(r['total_emissions'] for r in sample_results)
    summary = {
        'processing_date': datetime.now().isoformat(),
        'files_processed': len(sample_results),
        'total_emissions_kg_co2e': round(total_emissions, 2),
        'energy_breakdown': {
            'electricity': {'total_emissions': 492.0, 'count': 1},
            'natural_gas': {'total_emissions': 689.0, 'count': 1},
            'gasoline': {'total_emissions': 164.47, 'count': 1}
        },
        'document_types': {
            'electricity_bill': 1,
            'gas_bill': 1,
            'fuel_receipt': 1
        }
    }
    
    return sample_results, summary

def create_pdf_report(summary, detailed_results):
    """Generate PDF report (simplified version)"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        
        buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(buffer.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Carbon Emissions Audit Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Summary
        summary_text = f"""
        <b>Executive Summary</b><br/>
        Processing Date: {summary['processing_date'][:10]}<br/>
        Files Processed: {summary['files_processed']}<br/>
        Total Emissions: {summary['total_emissions_kg_co2e']} kg CO₂e<br/>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Table
        if summary['energy_breakdown']:
            table_data = [['Energy Source', 'Emissions (kg CO₂e)', 'Documents']]
            for energy_type, data in summary['energy_breakdown'].items():
                table_data.append([
                    energy_type.replace('_', ' ').title(),
                    f"{data['total_emissions']:.2f}",
                    str(data['count'])
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        doc.build(story)
        return buffer.name
    except ImportError:
        # If reportlab is not available, create a simple text file
        buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w')
        buffer.write(f"Carbon Emissions Report\n")
        buffer.write(f"Generated: {datetime.now()}\n")
        buffer.write(f"Total Emissions: {summary['total_emissions_kg_co2e']} kg CO₂e\n")
        buffer.write(f"Files Processed: {summary['files_processed']}\n")
        buffer.close()
        return buffer.name

def main():
    st.set_page_config(
        page_title="AI Carbon Accounting System",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 AI Industrial Carbon Accounting")
    st.markdown("**Automated emissions tracking and compliance reporting**")
    
    # Initialize session state
    if 'results_loaded' not in st.session_state:
        st.session_state['results_loaded'] = False
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Upload & Process", "View Results", "Generate Reports"])
    
    # Load sample data button
    if st.sidebar.button("🎯 Load Sample Data (Quick Demo)"):
        results, summary = create_sample_data()
        st.session_state['results'] = results
        st.session_state['summary'] = summary
        st.session_state['results_loaded'] = True
        st.sidebar.success("✅ Sample data loaded!")
    
    if page == "Upload & Process":
        st.header("📁 Document Processing")
        
        # Check if we have existing results
        if os.path.exists('outputs/fast_extraction_results.json'):
            with st.expander("📊 Load Previous Results"):
                if st.button("Load Previous Processing Results"):
                    try:
                        with open('outputs/fast_extraction_results.json', 'r') as f:
                            data = json.load(f)
                            st.session_state['results'] = data['detailed_results']
                            st.session_state['summary'] = data['summary']
                            st.session_state['results_loaded'] = True
                        st.success("✅ Previous results loaded!")
                    except Exception as e:
                        st.error(f"Error loading results: {e}")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload energy bills, invoices, or operational records",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("🚀 Process Documents", type="primary"):
                # Create temporary directory
                temp_dir = tempfile.mkdtemp()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Save uploaded files
                    for i, uploaded_file in enumerate(uploaded_files):
                        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
                        status_text.text(f"Saving {uploaded_file.name}...")
                    
                    # Process files
                    status_text.text("Processing documents with AI...")
                    extractor = CloudCarbonExtractor(use_mock=True)
                    results = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        text = extractor.extract_text_from_image(file_path)
                        result = extractor.smart_extract_energy_data(text, uploaded_file.name)
                        results.append(result)
                        
                        progress_bar.progress((len(uploaded_files) + i + 1) / (len(uploaded_files) * 2))
                        status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Generate summary
                    summary = {
                        'processing_date': datetime.now().isoformat(),
                        'files_processed': len(results),
                        'total_emissions_kg_co2e': round(sum(r['total_emissions'] for r in results), 2),
                        'energy_breakdown': {},
                        'document_types': {}
                    }
                    
                    # Calculate breakdowns
                    for result in results:
                        doc_type = result.get('document_type', 'unknown')
                        summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
                        
                        for energy_type, data in result['extracted_data'].items():
                            if energy_type not in summary['energy_breakdown']:
                                summary['energy_breakdown'][energy_type] = {'total_emissions': 0, 'count': 0}
                            summary['energy_breakdown'][energy_type]['total_emissions'] += data['emissions_kg_co2e']
                            summary['energy_breakdown'][energy_type]['count'] += 1
                    
                    # Save to session state
                    st.session_state['results'] = results
                    st.session_state['summary'] = summary
                    st.session_state['results_loaded'] = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    st.success("✅ Processing complete!")
                    st.balloons()
                    
                    # Show quick results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Processed", summary['files_processed'])
                    with col2:
                        st.metric("Total Emissions", f"{summary['total_emissions_kg_co2e']:.2f} kg CO₂e")
                    with col3:
                        st.metric("Energy Sources", len(summary['energy_breakdown']))
                        
                except Exception as e:
                    st.error(f"Error processing files: {e}")
                    st.write("Please try with the sample data first.")
    
    elif page == "View Results":
        st.header("📊 Results Dashboard")
        
        if not st.session_state.get('results_loaded') or 'summary' not in st.session_state:
            st.warning("⚠️ No data available. Please:")
            st.write("1. Click 'Load Sample Data' in the sidebar for a quick demo, OR")
            st.write("2. Go to 'Upload & Process' to upload your own documents")
            return
        
        summary = st.session_state['summary']
        results = st.session_state['results']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emissions", f"{summary['total_emissions_kg_co2e']:.2f} kg CO₂e")
        with col2:
            st.metric("Files Processed", summary['files_processed'])
        with col3:
            st.metric("Energy Sources", len(summary['energy_breakdown']))
        with col4:
            st.metric("Processing Date", summary['processing_date'][:10])
        
        # Visualizations
        if summary['energy_breakdown']:
            st.subheader("📈 Emissions Visualization")
            
            # Prepare data for plotting
            energy_types = list(summary['energy_breakdown'].keys())
            emissions = [data['total_emissions'] for data in summary['energy_breakdown'].values()]
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    values=emissions,
                    names=[name.replace('_', ' ').title() for name in energy_types],
                    title="Emissions Distribution by Source"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(
                    x=[name.replace('_', ' ').title() for name in energy_types],
                    y=emissions,
                    title="Emissions by Energy Source",
                    labels={'x': 'Energy Source', 'y': 'Emissions (kg CO₂e)'},
                    color=emissions,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Extraction Results")
        
        # Convert results to DataFrame
        display_data = []
        for result in results:
            row = {
                'Filename': result['filename'],
                'Document Type': result.get('document_type', 'unknown').replace('_', ' ').title(),
                'Total Emissions (kg CO₂e)': f"{result['total_emissions']:.2f}",
                'Processing Date': result['extraction_date'][:10]
            }
            
            # Add energy-specific data
            for energy_type, data in result['extracted_data'].items():
                row[f"{energy_type.replace('_', ' ').title()}"] = f"{data['amount']:.2f} {data['unit']}"
            
            display_data.append(row)
        
        if display_data:
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)
        
        # Raw JSON data (expandable)
        with st.expander("🔍 View Raw Extraction Data"):
            st.json(results)
    
    elif page == "Generate Reports":
        st.header("📋 Generate Reports")
        
        if not st.session_state.get('results_loaded') or 'summary' not in st.session_state:
            st.warning("⚠️ No data available. Please process documents first or load sample data.")
            return
        
        summary = st.session_state['summary']
        results = st.session_state['results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 PDF Report")
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_path = create_pdf_report(summary, results)
                        
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_file.read(),
                                file_name=f"carbon_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        st.success("✅ PDF generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")
        
        with col2:
            st.subheader("📊 Excel Export")
            if st.button("Generate Excel Report"):
                try:
                    # Create Excel data
                    excel_data = []
                    for result in results:
                        for energy_type, data in result['extracted_data'].items():
                            excel_data.append({
                                'Filename': result['filename'],
                                'Document_Type': result.get('document_type', 'unknown'),
                                'Energy_Type': energy_type,
                                'Amount': data['amount'],
                                'Unit': data['unit'],
                                'Emissions_kg_CO2e': data['emissions_kg_co2e'],
                                'Processing_Date': result['extraction_date'][:10]
                            })
                    
                    if excel_data:
                        df_excel = pd.DataFrame(excel_data)
                        
                        # Convert to CSV (simpler than Excel)
                        csv = df_excel.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV Report",
                            data=csv,
                            file_name=f"carbon_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        st.success("✅ CSV generated successfully!")
                    else:
                        st.warning("No data to export")
                except Exception as e:
                    st.error(f"Error generating CSV: {e}")
        
        # Show summary
        st.subheader("📋 Processing Summary")
        
        # Summary metrics in a nice format
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.info(f"""
            **📊 Processing Statistics**
            - Files Processed: {summary['files_processed']}
            - Processing Date: {summary['processing_date'][:10]}
            - Total Sources: {len(summary['energy_breakdown'])}
            """)
        
        with summary_col2:
            st.success(f"""
            **🌱 Emissions Summary**
            - Total: {summary['total_emissions_kg_co2e']} kg CO₂e
            - Equivalent Trees: {int(summary['total_emissions_kg_co2e'] / 22):.0f}*
            - Car Miles: {int(summary['total_emissions_kg_co2e'] * 2.5):.0f}*
            """)
        
        with summary_col3:
            st.warning(f"""
            **📋 Document Types**
            {chr(10).join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in summary.get('document_types', {}).items()])}
            """)
        
        st.caption("*Approximate equivalents for reference")
        
        # JSON export
        with st.expander("📄 Export Raw JSON Data"):
            json_str = json.dumps({'summary': summary, 'detailed_results': results}, indent=2)
            st.download_button(
                label="📥 Download JSON Data",
                data=json_str,
                file_name=f"carbon_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()