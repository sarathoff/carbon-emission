# Fast Carbon Accounting Pipeline - Day 1 Implementation
import pytesseract
import re
import pandas as pd
from PIL import Image
import os
from datetime import datetime
import json

class FastCarbonExtractor:
    def __init__(self):
        # EPA Standard Emission Factors (kg CO2e)
        self.emission_factors = {
            'electricity': 0.4,  # per kWh
            'natural_gas': 5.3,  # per therm
            'gasoline': 8.89,    # per gallon
            'diesel': 10.21,     # per gallon
            'coal': 2.23,        # per lb
            'propane': 5.75      # per gallon
        }
        
        # Regex patterns for common energy units
        self.patterns = {
            'electricity': r'(\d+\.?\d*)\s*(kWh|kwh|kilowatt|KWH)',
            'natural_gas': r'(\d+\.?\d*)\s*(therm|therms|ccf|cubic feet)',
            'gasoline': r'(\d+\.?\d*)\s*(gal|gallon|gallons)',
            'diesel': r'(\d+\.?\d*)\s*(gal|gallon|gallons).*diesel',
            'amount': r'\$?(\d+\.?\d*)',
            'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        }
    
    def extract_text_from_image(self, image_path):
        """Fast OCR extraction"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"OCR Error for {image_path}: {e}")
            return ""
    
    def extract_energy_data(self, text, filename):
        """Extract energy consumption data"""
        results = {
            'filename': filename,
            'extraction_date': datetime.now().isoformat(),
            'raw_text': text[:200] + "...",  # First 200 chars for debugging
            'extracted_data': {},
            'total_emissions': 0
        }
        
        # Extract each energy type
        for energy_type, pattern in self.patterns.items():
            if energy_type in ['amount', 'date']:
                continue
                
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the largest number found (usually total consumption)
                amounts = [float(match[0]) for match in matches if match[0]]
                if amounts:
                    max_amount = max(amounts)
                    results['extracted_data'][energy_type] = {
                        'amount': max_amount,
                        'unit': matches[0][1] if matches[0][1] else 'unknown',
                        'emissions_kg_co2e': max_amount * self.emission_factors.get(energy_type, 0)
                    }
                    results['total_emissions'] += max_amount * self.emission_factors.get(energy_type, 0)
        
        # Extract dates and amounts for context
        date_matches = re.findall(self.patterns['date'], text)
        amount_matches = re.findall(self.patterns['amount'], text)
        
        if date_matches:
            results['extracted_data']['dates'] = date_matches[:3]  # First 3 dates found
        if amount_matches:
            amounts = [float(m) for m in amount_matches if float(m) > 0]
            results['extracted_data']['monetary_amounts'] = amounts[:5]  # First 5 amounts
        
        return results
    
    def process_directory(self, input_dir):
        """Process all images in directory"""
        results = []
        supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                print(f"Processing: {filename}")
                image_path = os.path.join(input_dir, filename)
                
                # Extract text
                text = self.extract_text_from_image(image_path)
                
                # Extract energy data
                if text.strip():
                    result = self.extract_energy_data(text, filename)
                    results.append(result)
                    
                    # Print immediate results
                    print(f"  Found emissions: {result['total_emissions']:.2f} kg CO2e")
                    for energy_type, data in result['extracted_data'].items():
                        if isinstance(data, dict) and 'amount' in data:
                            print(f"    {energy_type}: {data['amount']} {data['unit']}")
        
        return results
    
    def generate_summary_report(self, results):
        """Generate quick summary"""
        total_emissions = sum(r['total_emissions'] for r in results)
        processed_files = len(results)
        
        # Create summary by energy type
        energy_summary = {}
        for result in results:
            for energy_type, data in result['extracted_data'].items():
                if isinstance(data, dict) and 'emissions_kg_co2e' in data:
                    if energy_type not in energy_summary:
                        energy_summary[energy_type] = {'total_amount': 0, 'total_emissions': 0}
                    energy_summary[energy_type]['total_amount'] += data['amount']
                    energy_summary[energy_type]['total_emissions'] += data['emissions_kg_co2e']
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'files_processed': processed_files,
            'total_emissions_kg_co2e': round(total_emissions, 2),
            'energy_breakdown': energy_summary,
            'compliance_status': 'PRELIMINARY' if total_emissions > 0 else 'NO_DATA_FOUND'
        }
        
        return summary

# USAGE EXAMPLE - RUN THIS IMMEDIATELY
if __name__ == "__main__":
    # Initialize extractor
    extractor = FastCarbonExtractor()
    
    # Create necessary directories
    os.makedirs("data/raw_images", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Process sample data (replace with your image directory)
    input_directory = "data/raw_images"  # Create this folder and add sample bills
    
    if os.path.exists(input_directory):
        print("🚀 Starting Fast Carbon Extraction...")
        
        # Process all images
        results = extractor.process_directory(input_directory)
        
        # Generate summary
        summary = extractor.generate_summary_report(results)
        
        # Save results
        with open('outputs/extraction_results.json', 'w') as f:
            json.dump({'summary': summary, 'detailed_results': results}, f, indent=2)
        
        # Print summary
        print("\n📊 EXTRACTION SUMMARY:")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Total emissions: {summary['total_emissions_kg_co2e']} kg CO2e")
        print(f"Status: {summary['compliance_status']}")
        
        print("\n📋 Energy Breakdown:")
        for energy_type, data in summary['energy_breakdown'].items():
            print(f"  {energy_type}: {data['total_emissions']:.2f} kg CO2e")
            
        print(f"\n💾 Results saved to: outputs/extraction_results.json")
        
    else:
        print(f"❌ Directory '{input_directory}' not found!")
        print("Create the directory and add sample energy bills/invoices.")
        print("You can download samples from utility company websites.")