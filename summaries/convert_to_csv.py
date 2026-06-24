import os
import csv
import regex as re

# Define the path to your two folders
model_folder, system_folder = "summaries/model_summaries", "summaries/system_summaries"

gov_dataset = "GovReport"
output_csv = "summaries/summaries.csv"

with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    csvwriter = csv.writer(outfile)
    # Write headers
    csvwriter.writerow(["Model Summary", "System Summary", "Dataset"])

    for model_filename, system_filename in zip(os.listdir(model_folder), os.listdir(system_folder)):
        if model_filename.endswith('.txt') and system_filename.endswith('.txt'):
            model_file_path = os.path.join(model_folder, model_filename)
            system_file_path = os.path.join(system_folder, system_filename)

            with open(model_file_path, 'r', encoding='utf-8') as txtfile:
                model_content = txtfile.read()
            with open(system_file_path, 'r', encoding='utf-8') as txtfile:
                system_content = txtfile.read()

            csvwriter.writerow([model_content, system_content, gov_dataset])
    
    
    # for folder in folders:
    #     # Loop through all files in the directory


    #     for filename in os.listdir(folder):
    #         if filename.endswith('.txt'):
    #             file_path = os.path.join(folder, filename)

    #             with open(file_path, 'r', encoding='utf-8') as txtfile:
    #                 content = txtfile.read()

    #             # Write row with folder name, file name, and text content
    #             csvwriter.writerow([os.path.basename(folder), filename, gov_dataset])
