# synthetic_data_generator.py - Generate fake data to test pipeline IMMEDIATELY
import random
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime, timedelta

class SyntheticBillGenerator:
    def __init__(self):
        self.utilities = ['PowerCorp Electric', 'GasCo Natural Gas', 'FuelStation LLC']
        self.addresses = ['123 Main St', '456 Oak Ave', '789 Pine Rd']
        
    def generate_electricity_bill(self, filename):
        """Generate fake electricity bill image"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fallback to default if not available
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Header
        draw.text((50, 30), "ELECTRIC UTILITY COMPANY", fill='black', font=font_large)
        draw.text((50, 60), "Monthly Energy Statement", fill='black', font=font_medium)
        
        # Account info
        draw.text((50, 120), f"Account: {random.randint(100000, 999999)}", fill='black', font=font_small)
        draw.text((50, 140), f"Service Address: {random.choice(self.addresses)}", fill='black', font=font_small)
        
        # Billing period
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        draw.text((50, 180), f"Billing Period: {start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}", fill='black', font=font_small)
        
        # Energy usage - THE KEY DATA
        kwh_used = random.randint(500, 2000)
        draw.text((50, 220), "ELECTRICITY USAGE SUMMARY", fill='black', font=font_medium)
        draw.text((50, 250), f"Total kWh Used: {kwh_used} kWh", fill='black', font=font_medium)
        draw.text((50, 280), f"Previous Reading: {random.randint(10000, 50000)}", fill='black', font=font_small)
        draw.text((50, 300), f"Current Reading: {random.randint(10000, 50000)}", fill='black', font=font_small)
        
        # Charges
        rate = round(random.uniform(0.08, 0.15), 4)
        amount = round(kwh_used * rate, 2)
        draw.text((50, 340), f"Energy Charge: {kwh_used} kWh × ${rate}/kWh = ${amount}", fill='black', font=font_small)
        draw.text((50, 380), f"Total Amount Due: ${amount + random.randint(5, 25)}", fill='black', font=font_medium)
        
        img.save(filename)
        return kwh_used
    
    def generate_gas_bill(self, filename):
        """Generate fake natural gas bill"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Header
        draw.text((50, 30), "NATURAL GAS COMPANY", fill='black', font=font_large)
        draw.text((50, 60), "Gas Service Statement", fill='black', font=font_medium)
        
        # Account info
        draw.text((50, 120), f"Account: {random.randint(100000, 999999)}", fill='black', font=font_small)
        draw.text((50, 140), f"Service Address: {random.choice(self.addresses)}", fill='black', font=font_small)
        
        # Gas usage
        therms_used = random.randint(50, 300)
        draw.text((50, 220), "NATURAL GAS USAGE", fill='black', font=font_medium)
        draw.text((50, 250), f"Gas Used: {therms_used} Therms", fill='black', font=font_medium)
        draw.text((50, 280), f"Heating Degree Days: {random.randint(400, 800)}", fill='black', font=font_small)
        
        # Charges
        rate = round(random.uniform(0.8, 1.5), 3)
        amount = round(therms_used * rate, 2)
        draw.text((50, 340), f"Gas Charge: {therms_used} Therms × ${rate}/Therm = ${amount}", fill='black', font=font_small)
        
        img.save(filename)
        return therms_used
    
    def generate_fuel_receipt(self, filename):
        """Generate fake fuel receipt"""
        img = Image.new('RGB', (600, 800), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_large = ImageFont.truetype("arial.ttf", 20)
            font_medium = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Header
        draw.text((50, 30), "FUEL STATION", fill='black', font=font_large)
        draw.text((50, 60), "Receipt", fill='black', font=font_medium)
        
        # Date and time
        draw.text((50, 120), f"Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}", fill='black', font=font_small)
        
        # Fuel info
        gallons = round(random.uniform(8, 25), 2)
        fuel_type = random.choice(['Regular Gasoline', 'Diesel', 'Premium'])
        price_per_gallon = round(random.uniform(2.5, 4.0), 3)
        total = round(gallons * price_per_gallon, 2)
        
        draw.text((50, 180), f"Product: {fuel_type}", fill='black', font=font_medium)
        draw.text((50, 210), f"Gallons: {gallons}", fill='black', font=font_medium)
        draw.text((50, 240), f"Price/Gal: ${price_per_gallon}", fill='black', font=font_small)
        draw.text((50, 270), f"Total: ${total}", fill='black', font=font_medium)
        
        img.save(filename)
        return gallons, fuel_type

def generate_sample_dataset():
    """Generate complete sample dataset in 30 seconds"""
    generator = SyntheticBillGenerator()
    
    # Create directories
    os.makedirs("data/raw_images", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("🎯 Generating synthetic energy documents...")
    
    # Generate 10 sample documents
    expected_emissions = 0
    
    for i in range(3):
        # Electricity bills
        kwh = generator.generate_electricity_bill(f"data/raw_images/electric_bill_{i+1}.png")
        expected_emissions += kwh * 0.4  # 0.4 kg CO2e per kWh
        print(f"Generated electric_bill_{i+1}.png - {kwh} kWh")
    
    for i in range(3):
        # Gas bills
        therms = generator.generate_gas_bill(f"data/raw_images/gas_bill_{i+1}.png")
        expected_emissions += therms * 5.3  # 5.3 kg CO2e per therm
        print(f"Generated gas_bill_{i+1}.png - {therms} therms")
    
    for i in range(2):
        # Fuel receipts
        gallons, fuel_type = generator.generate_fuel_receipt(f"data/raw_images/fuel_receipt_{i+1}.png")
        expected_emissions += gallons * 8.89  # Approximate for gasoline
        print(f"Generated fuel_receipt_{i+1}.png - {gallons} gallons {fuel_type}")
    
    print(f"\n✅ Generated 8 sample documents")
    print(f"📊 Expected total emissions: ~{expected_emissions:.2f} kg CO₂e")
    print(f"📁 Files saved to: data/raw_images/")
    print("\n🚀 Now run: python carbon_pipeline_day1.py")

if __name__ == "__main__":
    generate_sample_dataset()