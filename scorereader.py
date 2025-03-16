import os
from pathlib import Path
from typing import List, Dict
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter
import json
import csv
from datetime import datetime
import io

# Load environment variables
load_dotenv()

class ScoreRecord:
    def __init__(self, rank: int, commander_name: str, alliance_name: str, points: int):
        self.rank = rank
        self.commander_name = self.normalize_name(commander_name)
        self.points = points
        
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize commander names to handle slight variations and OCR mistakes"""
        # Convert to lowercase for comparison
        normalized = name.lower()
        
        # Extended special characters mapping
        special_chars = {
            # Latin characters
            'ƞ': 'n', 'ạ': 'a', 'ń': 'n', 'ñ': 'n', 'ã': 'a', 'ā': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 'ē': 'e',
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'å': 'a',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i', 'ī': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'ō': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u', 'ū': 'u',
            # Cyrillic characters that might be confused with Latin
            'ѵ': 'v', 'ԅ': 'n', 'а': 'a', 'е': 'e', 'о': 'o',
            # Common OCR confusions
            #'ph': 'f', '0': 'o', '1': 'l', '5': 's',
            # Additional special characters
            'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ø': 'o'
        }
        
        # First pass: replace special characters
        for special, normal in special_chars.items():
            normalized = normalized.replace(special, normal)
        
        # Remove any remaining non-alphanumeric characters except spaces
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common words that might be inconsistent
        words_to_remove = ['the', 'of', 'and', 'or', 'in', 'at', 'to']
        normalized_words = normalized.split()
        normalized_words = [word for word in normalized_words if word not in words_to_remove]
        normalized = ' '.join(normalized_words)
        
        return normalized
        
    def __eq__(self, other):
        if not isinstance(other, ScoreRecord):
            return False
        return (self.rank == other.rank and 
                self.normalize_name(self.commander_name) == self.normalize_name(other.commander_name) and 
                self.points == other.points)
    
    def __hash__(self):
        return hash((self.rank, self.normalize_name(self.commander_name), self.points))

class ScoreReader:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def encode_image(self, image_path: str) -> str:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def preprocess_image(self, image_path: str) -> str:
        """Preprocess image to improve OCR accuracy and return base64 string"""
        with Image.open(image_path) as img:
            # Convert to RGB mode if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Get original size
            original_width, original_height = img.size
            
            # Upscale if image is too small (minimum 1500px width)
            min_width = 1500
            if original_width < min_width:
                scale_factor = min_width / original_width
                new_size = (min_width, int(original_height * scale_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.6)  # Increase contrast by 50%

            # Sharpen the image
            img = img.filter(ImageFilter.SHARPEN)
            img = img.filter(ImageFilter.DETAIL)  # Enhance details

            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte_arr).decode('utf-8')

    def process_image(self, image_path: str) -> List[ScoreRecord]:
        print(f"\nProcessing image: {image_path}")
        
        # Preprocess and encode the image
        try:
            print("Preprocessing image...")
            base64_image = self.preprocess_image(image_path)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return []
        
        # Prepare the message for GPT-4 Vision
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this game screenshot and extract the rankings. Pay special attention to commander names, ensuring exact character recognition including special characters. Return ONLY raw JSON data in this exact format, with NO markdown:\n"
                               "{\n"
                               "  \"records\": [\n"
                               "    {\n"
                               "      \"rank\": <integer>,\n"
                               "      \"commander_name\": \"<string>\",\n"
                               "      \"alliance_name\": \"[DRp] Dr Pepper Fresh\",\n"
                               "      \"points\": <integer>\n"
                               "    }\n"
                               "  ]\n"
                               "}\n"
                               "Important:\n"
                               "1. Ensure commander names are exactly as shown, preserving all special characters\n"
                               "2. Double-check any ambiguous characters (0/O, l/1, etc.)\n"
                               "3. Pay attention to diacritical marks and special characters in names\n"
                               "4. Verify numbers are correctly distinguished from letters"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        print("Sending to GPT-4 o...")
        # Get response from GPT-4 Vision
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )

        try:
            # Parse the JSON response
            print("Parsing response...")
            response_content = response.choices[0].message.content
            
            # Clean up the response content by removing markdown formatting
            response_content = response_content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:]  # Remove ```json prefix
            if response_content.startswith("```"):
                response_content = response_content[3:]  # Remove ``` prefix
            if response_content.endswith("```"):
                response_content = response_content[:-3]  # Remove ``` suffix
            response_content = response_content.strip()
            
            # print(f"Raw response (cleaned): {response_content}")  # Debug output
            data = json.loads(response_content)
            records = []
            for record in data['records']:
                score_record = ScoreRecord(
                    rank=record['rank'],
                    commander_name=record['commander_name'],
                    alliance_name=record['alliance_name'],
                    points=record['points']
                )
                records.append(score_record)
            print(f"Successfully extracted {len(records)} records from {os.path.basename(image_path)}")
            return records
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing image {image_path}: {str(e)}")
            print(f"Response content that failed to parse: {response_content}")  # Debug output
            return []

    def process_folder(self, folder_path: str) -> List[ScoreRecord]:
        all_records = set()  # Using a set for automatic deduplication
        folder = Path(folder_path)
        
        # Get list of all image files
        image_files = list(folder.glob('*.png')) + list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg'))
        
        if not image_files:
            print(f"\nNo image files found in {folder_path}")
            print("Supported formats: .png, .jpg, .jpeg")
            return []
            
        print(f"\nFound {len(image_files)} images to process")
        
        # Process each image in the folder
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"\nProcessing image {i} of {len(image_files)}")
                records = self.process_image(str(image_path))
                previous_count = len(all_records)
                all_records.update(records)
                new_records = len(all_records) - previous_count
                print(f"Added {new_records} new unique records (filtered out {len(records) - new_records} duplicates)")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        return list(all_records)

def process_folder_type(reader: ScoreReader, folder_path: str, folder_type: str) -> List[ScoreRecord]:
    if not folder_path:
        print(f"Skipping {folder_type} folder processing...")
        return []

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Process images
    print(f"\nProcessing {folder_type} images in {folder_path}...")
    records = reader.process_folder(folder_path)
    
    # Sort records by rank
    records.sort(key=lambda x: x.rank)
    return records

def print_records(records: List[ScoreRecord], category: str):
    if not records:
        print(f"\nNo {category} records to display.")
        return
        
    print(f"\n{category} Records:")
    print("-" * 80)
    for record in records:
        print(f"Rank: {record.rank}, Commander: {record.commander_name}, Points: {record.points}")
    print("-" * 80)

def create_combined_csv(vs_records: List[ScoreRecord], 
                       donation_records: List[ScoreRecord], 
                       kill_day_records: List[ScoreRecord]):
    # Create a dictionary to store all commanders and their records
    commanders = {}
    
    def add_or_update_commander(record: ScoreRecord, category: str):
        normalized_name = ScoreRecord.normalize_name(record.commander_name)
        if normalized_name not in commanders:
            # Initialize with empty values
            commanders[normalized_name] = {
                'commander': record.commander_name,  # Use original name for display
                'vsrank': '',
                'vspoints': '',
                'donationsrank': '',
                'donationspoints': '',
                'killdayrank': '',
                'killdaypoints': ''
            }
        # Update the specific category's values
        if category == 'vs':
            commanders[normalized_name].update({
                'vsrank': record.rank,
                'vspoints': record.points
            })
        elif category == 'donations':
            commanders[normalized_name].update({
                'donationsrank': record.rank,
                'donationspoints': record.points
            })
        elif category == 'killday':
            commanders[normalized_name].update({
                'killdayrank': record.rank,
                'killdaypoints': record.points
            })
    
    # Process all records using the new helper function
    for record in vs_records:
        add_or_update_commander(record, 'vs')
    
    for record in donation_records:
        add_or_update_commander(record, 'donations')
    
    for record in kill_day_records:
        add_or_update_commander(record, 'killday')

    # Create the CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"combined_scores_{timestamp}.csv"
    
    # Write to CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['commander', 'vsrank', 'vspoints', 'donationsrank', 
                     'donationspoints', 'killdayrank', 'killdaypoints']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        # Sort by commander name for consistent output
        for commander in sorted(commanders.keys()):
            writer.writerow(commanders[commander])
    
    return csv_filename

def create_email_summary(csv_filename: str, client: OpenAI) -> str:
    """Generate an email summary of the top 20 performers in each category."""
    # Read the CSV file
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        csv_content = csvfile.read()

    prompt = f"""Create an alliance email summarizing the top performers from this CSV data.
The CSV contains rankings and points for VS scores, Donations, and Kill Day scores.
Format the email as follows:

Congratulations to the top performers this week.

Top 20 VS Weekly Scorers:
[List only top 20, format: Name - Points]

Top 20 Donors:
[List only top 20, format: Name - Points]

Top 20 Kill Day Scores:
[List only top 20, format: Name - Points]

CSV Data:
{csv_content}

Important:
1. Only include commanders who have scores in each category
2. Format numbers with commas for readability
3. Keep the tone congratulatory and positive
4. Skip any category that has no data
"""

    print("\nGenerating email summary...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    return response.choices[0].message.content

def main():
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    # Get all folder paths upfront
    print("Enter folder paths for each category (press Enter to skip):")
    vs_path = input("VS screenshots folder path: ")
    donation_path = input("Donation screenshots folder path: ")
    kill_day_path = input("Kill Day screenshots folder path: ")

    reader = ScoreReader()
    
    # Process each type of folder
    vs_records = process_folder_type(reader, vs_path, "VS")
    donation_records = process_folder_type(reader, donation_path, "Donation")
    kill_day_records = process_folder_type(reader, kill_day_path, "Kill Day")

    # Print results for each category
    print("\n=== Final Results ===")
    print_records(vs_records, "VS")
    print_records(donation_records, "Donation")
    print_records(kill_day_records, "Kill Day")

    # Create combined CSV report
    if any([vs_records, donation_records, kill_day_records]):  # Only create CSV if we have any records
        csv_filename = create_combined_csv(vs_records, donation_records, kill_day_records)
        print(f"\nCombined results have been saved to: {csv_filename}")
        
        # Generate and save email summary
        email_content = create_email_summary(csv_filename, reader.client)
        email_filename = f"email_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(email_filename, 'w', encoding='utf-8') as f:
            f.write(email_content)
        print(f"\nEmail summary has been saved to: {email_filename}")
        print("\nEmail Content Preview:")
        print("=" * 80)
        print(email_content)
        print("=" * 80)
    else:
        print("\nNo records were processed, skipping CSV and email creation.")

if __name__ == "__main__":
    main()
