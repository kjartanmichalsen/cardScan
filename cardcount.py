import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Load the JSON data from the file in the output/ folder
with open('output/my_cards.json', 'r') as file:
    cards = json.load(file)

# Create a dictionary to count the number of each card
card_count = {}

for card in cards:
    card_key = (card['name'], card['set_code'], card['card_number'], card['types_or_trainer_type'])
    if card_key in card_count:
        card_count[card_key] += 1
    else:
        card_count[card_key] = 1

# Create a DataFrame from the dictionary
df = pd.DataFrame([
    {'count': count, 'name': name, 'set_code': set_code, 'card_number': card_number, 'type': type_}
    for (name, set_code, card_number, type_), count in card_count.items()
])

# Sort the DataFrame by type alphabetically
df_sorted = df.sort_values(by='type')

# Create an Excel workbook and worksheet using openpyxl
wb = Workbook()
ws = wb.active

# Write the DataFrame to the worksheet
for row in dataframe_to_rows(df_sorted, index=False, header=True):
    ws.append(row)

# Save the workbook to an Excel file
wb.save('output/card_counts.xlsx')

print("The card counts have been written to 'output/card_counts.xlsx'.")
