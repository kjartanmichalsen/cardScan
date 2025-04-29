import os
import cv2
import numpy as np
import re
import subprocess
import json
import csv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import openpyxl
import time
from playsound import playsound
# Mac error install fix: pip install playsound@git+https://github.com/taconi/playsound
# then pip3 install PyObjC

# setup env: python3 -m venv <path>/cardScan
# env: source <path>/cardScan/bin/activate
# pip3 install azure-ai-vision-imageanalysis  
# pip3 install opencv-python
# export VISION_KEY=<your_key>
# export VISION_ENDPOINT=<your endpoint>

# Function to find and crop the Pokémon card
def find_and_crop_card(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    
    # Ensure the coordinates are within the image dimensions
    height, width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    
    cropped_card = image[y:y+h, x:x+w]
    cropped_image_path = 'cropped_card.jpg'
    cv2.imwrite(cropped_image_path, cropped_card)
    return cropped_image_path

# Function to use Azure OCR to extract text
def extract_text_from_image(image_path):
    subscription_key = os.getenv('VISION_KEY')
    endpoint = os.getenv('VISION_ENDPOINT')
    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(subscription_key))
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    result = client.analyze(image_data=image_data, visual_features=[VisualFeatures.READ])
    text = ""
    if result.read is not None and result.read.blocks:
        for line in result.read.blocks[0].lines:
            text += line.text + " "
    else:
        print("No text blocks found in the image.")
        return text.strip(), [], None, None, None
    pattern_combinations = re.compile(r'\b(SVI|PAL|OBF|MEW|PAR|PAF|TEF|TWM|SFA|SCR|SSP|PRE|JTG)\b')
    matches_combinations = pattern_combinations.findall(text)
    pattern_card_number = re.compile(r'\b(\d{3}/\d{3})\b')
    match_card_number = pattern_card_number.search(text)
    if match_card_number:
        card_number = match_card_number.group()
        card_id = card_number.split('/')[0]
    else:
        card_number = None
        card_id = None
    setId_mapping = {
        'PRE': 'sv08.5',
        'PAR': 'sv04',
        'MEW': 'sv03.5',
        'JTG': 'sv09',
        'OBF': 'sv03',
        'PAL': 'sv02',
        'PAF': 'sv04.5',
        'SVP': 'svp',
        'SVI': 'sv01',
        'SFA': 'sv06.5',
        'SCR': 'sv07',
        'SSP': 'sv08',
        'TEF': 'sv05',
        'TWM': 'sv06'
    }
    if matches_combinations:
        setCode = matches_combinations[-1]
        setId = setId_mapping.get(setCode, None)
    else:
        setId = None
        print("No occurrences of the specified three-letter combinations found.")
        print("Extracted Text:", text.strip())
    return text.strip(), matches_combinations, card_number, card_id, setId

# Function to query the TCGdex API using curl and extract relevant variables
def query_tcgdex_api(setId, cardId):
    if setId and cardId:
        url = f"https://api.tcgdex.net/v2/en/cards/{setId}-{cardId}"
        response = subprocess.run(['curl', '-s', url], capture_output=True, text=True)
        api_response = json.loads(response.stdout)
        name = api_response.get('name')
        category = api_response.get('category')
        id = api_response.get('id')
        image = api_response.get('image')
        if category == 'Pokemon':
            types = api_response.get('types')
            types_or_trainer_type = types[0] if types else None
            return name, category, id, image, types_or_trainer_type
        elif category == 'Trainer':
            trainerType = api_response.get('trainerType')
            return name, category, id, image, trainerType
        else:
            return name, category, id, image, None
    else:
        return None, None, None, None, None

# Function to append data to a CSV file
def append_to_csv(file_name, data):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Function to append data to a JSON file
def append_to_json(file_name, data):
    if os.path.exists(file_name):
        with open(file_name, mode='r') as file:
            existing_data = json.load(file)
            existing_data.append(data)
        with open(file_name, mode='w') as file:
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_name, mode='w') as file:
            json.dump([data], file, indent=4)

# Function to detect motion and capture image
def detect_motion_and_capture(skip_motion_detection=False):
    cap = cv2.VideoCapture(0)
    
    if not skip_motion_detection:
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        
        # Get the dimensions of the frame
        height, width = frame1.shape[:2]
        
        # Define the coordinates for the square in the center
        center_x, center_y = width // 2, height // 2
        roi_x1, roi_y1 = center_x - 50, center_y - 300
        roi_x2, roi_y2 = center_x + 50, center_y + 300
        
        while cap.isOpened():
            if cv2.waitKey(10) == 32:  # 32 is the ASCII code for the space bar
                print("Skipping motion detection due to space bar press.")
                ret, frame = cap.read()
                image_path = 'captured_image.jpg'
                print("Picture taken.")
                cv2.imwrite(image_path, frame)
                cap.release()
                return image_path
            
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 2000:  # Increase the minimum contour area to reduce sensitivity
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the contour is within the defined ROI
                if roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2:
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print("Motion detected! Waiting for 1.2 seconds for card to fall down...")
                    cv2.waitKey(1200)
                    ret, frame = cap.read()
                    image_path = 'captured_image.jpg'
                    cv2.imwrite(image_path, frame)
                    print("Motion detected picture taken.")
                    cap.release()
                    return image_path
            
            frame1 = frame2
            ret, frame2 = cap.read()
            cv2.imshow("Preview", frame1)
            
            if cv2.waitKey(10) == 27:  # 27 is the ASCII code for the ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return None
    else:
        print("_-_-_-_-_ Check camera - retry in 3 seconds _-_-_-_-_")
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            cv2.imshow("Preview", frame)
            if cv2.waitKey(10) == 27:  # 27 is the ASCII code for the ESC key
                break
        
        ret, frame = cap.read()
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame)
        print("Picture taken.")
        cap.release()
        cv2.destroyAllWindows()
        return image_path

# Function to append data to an Excel file
def append_to_excel(file_name, data):
    if os.path.exists(file_name):
        workbook = openpyxl.load_workbook(file_name)
        sheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        # Write the header row
        sheet.append(["Category", "Types or Trainer Type", "Name", "Set Code", "Card Number"])

    # Append the data row
    sheet.append(data)
    workbook.save(file_name)

# Main function
def main():
    skip_motion_detection = False
    for _ in range(100):
        print("------------- waiting for new card --------------")
        image_path = detect_motion_and_capture(skip_motion_detection=skip_motion_detection)
        if image_path:
            cropped_image_path = find_and_crop_card(image_path)
            extracted_text, matches_combinations, card_number, card_id, setId = extract_text_from_image(cropped_image_path)
            
            if not card_id or not matches_combinations:
                print(f"Skipping image {image_path}: No card ID or no occurrences of the three-letter uppercase combinations found.")
                skip_motion_detection = True
                playsound('error.wav')
                print("Extracted Text:", extracted_text)
                continue
            
            if not setId:
                print(f"Skipping image {image_path}: No set ID found.")
                skip_motion_detection = True
                playsound('error.wav')
                print("Extracted Text:", extracted_text)
                continue
            
            print("Extracted Text:", extracted_text)
            print("Matches:", matches_combinations)
            print("Card Number:", card_number)
            print("Card ID:", card_id)
            print("Set ID:", setId)
            
            name, category, id, image_url, types_or_trainer_type = query_tcgdex_api(setId, card_id)
            
            print("Category:", category)
            print("ID:", id)
            print("Image URL:", image_url)
            
            if category == 'Pokemon':
                print("Type:", types_or_trainer_type)
            elif category == 'Trainer':
                print("Trainer Type:", types_or_trainer_type)
            
            print("### Name:", name)

            setCode = matches_combinations[-1] if matches_combinations else None
            csv_data = [category, name, card_id, setCode, card_number, types_or_trainer_type, image_url]
            
            # Uncomment if you want CSV data
            # append_to_csv('output/my_cards.csv', csv_data)
            
            json_data = {
                "category": category,
                "name": name,
                "card_id": card_id,
                "set_code": setCode,
                "card_number": card_number,
                "types_or_trainer_type": types_or_trainer_type,
                "image_url": image_url,
                "tcgDexSet": setId
            }
            append_to_json('output/my_cards.json', json_data)
            
            # Append to Excel file
            excel_data = [category, types_or_trainer_type, name, setCode, card_number]
            append_to_excel('output/my_cards.xlsx', excel_data)
            
            # Play a happy sound
            playsound('ok.wav')

            # Reset skip_motion_detection for the next iteration
            skip_motion_detection = False
        else:
            break

if __name__ == "__main__":
    main()

