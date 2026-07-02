# from minicheck.minicheck import MiniCheck
# import nltk
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import csv

# # Open the file safely using a context manager
# with open('data.csv', mode='r', newline='', encoding='utf-8') as file:
#     # Create the reader object
#     csv_reader = csv.reader(file)
    
#     # Optional: Skip the header row if you don't want to process it
#     header = next(csv_reader)
#     print(f"Headers: {header}")
    
#     # Loop through each row
#     for row in csv_reader:
#         print(row)  # Each row is a list of strings

# nltk.download('punkt_tab')

# doc = "A group of students gather in the school library to study for their upcoming final exams."
# claim_1 = "The students are preparing for an examination."
# claim_2 = "The students are on vacation."

# # model_name can be one of the followings:
# # ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']

# #  MiniCheck-Flan-T5-Large (770M) is the best fack-checking model 
# # with size < 1B and reaches GPT-4 performance.
# scorer = MiniCheck(model_name='flan-t5-large', cache_dir='./ckpts')
# pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

# print(pred_label) # [1, 0]
# print(raw_prob)   # [0.9805923700332642, 0.007121330592781305]
