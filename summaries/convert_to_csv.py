import os
import csv
import regex as re

from pathlib import Path
import sys
THIS_FILE = Path(__file__).resolve()
SUMMARIES_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from summary_rewards.reward_func import reward_function_single

# Define the path to your two folders
model_folder, reference_folder = "summaries/model_summaries", "summaries/reference_summaries"

gov_dataset = "GovReport"
output_csv = "summaries/summaries.csv"

def main():
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        csvwriter = csv.writer(outfile)
        # Write headers
        csvwriter.writerow(["model_summary", "referenece_summary", "dataset", "reward"])

        for model_filename, reference_filename in zip(os.listdir(model_folder), os.listdir(reference_folder)):
            if model_filename.endswith('.txt') and reference_filename.endswith('.txt'):
                model_file_path = os.path.join(model_folder, model_filename)
                ref_file_path = os.path.join(reference_folder, reference_filename)

                with open(model_file_path, 'r', encoding='utf-8') as txtfile:
                    model_content = txtfile.read()
                with open(ref_file_path, 'r', encoding='utf-8') as txtfile:
                    ref_content = txtfile.read()

                reward_score = reward_function_single(model_content, ref_content)
                
                csvwriter.writerow([model_content, ref_content, gov_dataset, reward_score])


if __name__ == "__main__":
    main()

    
    
    # for folder in folders:
    #     # Loop through all files in the directory


    #     for filename in os.listdir(folder):
    #         if filename.endswith('.txt'):
    #             file_path = os.path.join(folder, filename)

    #             with open(file_path, 'r', encoding='utf-8') as txtfile:
    #                 content = txtfile.read()

    #             # Write row with folder name, file name, and text content
    #             csvwriter.writerow([os.path.basename(folder), filename, gov_dataset])
