import streamlit as st
import pytesseract
import re
import pandas as pd
from PIL import Image
import os
from datetime import datetime
import json
from transformers import pipeline

# Initialize a more powerful question-answering pipeline
@st.cache_resource
def get_qa_pipeline():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

qa_pipeline = get_qa_pipeline()

class FastCarbonExtractor:
    def __init__(self):
        self.emission_factors = {
            'electricity': 0.4, 'natural_gas': 5.3, 'gasoline': 8.89,
            'diesel': 10.21, 'coal': 2.23, 'propane': 5.75
        }
        self.patterns = {
            'electricity': r'(\d+\.?\d*)\s*(kWh|kwh|kilowatt|KWH)',
            'natural_gas': r'(\d+\.?\d*)\s*(therm|therms|ccf|cubic feet)',
            'gasoline': r'(\d+\.?\d*)\s*(gal|gallon|gallons)',
            'diesel': r'(\d+\.?\d*)\s*(gal|gallon|gallons).*diesel',
        }

    def extract_text_from_image(self, image):
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            st.error(f"OCR Error: {e}")
            return ""

    def extract_energy_data(self, text, filename):
        results = {
            'filename': filename, 'raw_text': text, 'extracted_data': {},
            'total_emissions': 0, 'extraction_date': datetime.now().isoformat()
        }
        for energy_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amounts = [float(match[0]) for match in matches if match[0]]
                if amounts:
                    max_amount = max(amounts)
                    emissions = max_amount * self.emission_factors.get(energy_type, 0)
                    results['extracted_data'][energy_type] = {
                        'amount': max_amount, 'unit': matches[0][1] or 'unknown',
                        'emissions_kg_co2e': emissions
                    }
                    results['total_emissions'] += emissions
        return results

def create_context_for_qa(extraction_results):
    """Create a rich, detailed context for each document for the QA model."""
    contexts = []
    for result in extraction_results:
        context = f"Document: {result['filename']}\n"
        context += f"Total Emissions: {result['total_emissions']:.2f} kg CO2e\n"
        if result['extracted_data']:
            context += "Extracted Data:\n"
            for energy_type, data in result['extracted_data'].items():
                context += f"- {energy_type.replace('_', ' ').title()}: {data['amount']} {data['unit']}\n"
        context += "---\n"
        context += result['raw_text']
        contexts.append(context)
    return "\n\n---\n\n".join(contexts)

def answer_from_structured_data(question, extraction_results):
    question = question.lower()
    # More robust check for total consumption questions
    if 'total' in question and ('consumption' in question or 'usage' in question):
        for energy_type in ['electricity', 'natural gas', 'gasoline', 'diesel']:
            if energy_type in question:
                total_consumption = sum(
                    res.get('extracted_data', {}).get(energy_type.replace(' ', '_'), {}).get('amount', 0)
                    for res in extraction_results
                )
                unit = next((
                    res.get('extracted_data', {}).get(energy_type.replace(' ', '_'), {}).get('unit')
                    for res in extraction_results if res.get('extracted_data', {}).get(energy_type.replace(' ', '_'))
                ), 'units')
                if total_consumption > 0:
                    return f"The total consumption for {energy_type} is {total_consumption:.2f} {unit}."

    if 'total' in question and 'emissions' in question:
        total_emissions = sum(r['total_emissions'] for r in extraction_results)
        return f"The total estimated emissions from all documents is {total_emissions:.2f} kg CO2e."
    
    return None

def main():
    st.title("Carbon Accounting Pipeline with Chat")
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload your energy bills (images)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = []
    if 'qa_context' not in st.session_state:
        st.session_state.qa_context = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if uploaded_files:
        extractor = FastCarbonExtractor()
        st.session_state.extraction_results = []
        st.session_state.messages = []
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                text = extractor.extract_text_from_image(image)
                if text.strip():
                    result = extractor.extract_energy_data(text, uploaded_file.name)
                    st.session_state.extraction_results.append(result)
            st.session_state.qa_context = create_context_for_qa(st.session_state.extraction_results)
        st.sidebar.success("Processing complete!")

    if st.session_state.extraction_results:
        st.header("Extracted Data")
        total_emissions = sum(r['total_emissions'] for r in st.session_state.extraction_results)
        st.metric("Total Estimated Emissions", f"{total_emissions:.2f} kg CO2e")

        display_data = []
        for result in st.session_state.extraction_results:
            st.sidebar.image(Image.open(next(f for f in uploaded_files if f.name == result['filename'])), caption=f"Uploaded: {result['filename']}", use_container_width=True)
            for energy_type, data in result['extracted_data'].items():
                if isinstance(data, dict) and 'amount' in data:
                    display_data.append({
                        "File": result['filename'],
                        "Energy Type": energy_type.replace('_', ' ').title(),
                        "Consumption": f"{data['amount']} {data['unit']}",
                        "Emissions (kg CO2e)": f"{data['emissions_kg_co2e']:.2f}"
                    })
        if display_data:
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)

        st.header("Chat About Your Data")
        st.info("Ask questions about your documents. The chatbot can perform calculations or find information in the text.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question, e.g., 'What is the total electricity consumption?'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_from_structured__data(prompt, st.session_state.extraction_results)
                    if answer is None and st.session_state.qa_context:
                        response = qa_pipeline(question=prompt, context=st.session_state.qa_context)
                        answer = response['answer'] if response['score'] > 0.1 else "I couldn\'t find a specific answer in the documents."
                    elif not st.session_state.qa_context:
                        answer = "No text available for searching. Please upload documents."
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("Upload one or more documents to begin analysis and chat.")

if __name__ == "__main__":
    main()

