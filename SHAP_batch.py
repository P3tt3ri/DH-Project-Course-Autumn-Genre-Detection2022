from datetime import datetime
 # record current timestamp
start = datetime.now()
print("started:", start)

import pandas as pd
import sys
from tqdm import tqdm
import json
import re
import torch
import numpy as np
from transformers import AutoTokenizer, BertConfig
from transformers import BertForSequenceClassification

def read_labels_set(labels_set_path):
    with open(labels_set_path, "r") as f:
        labels2id = json.load(f)
    return labels2id

labels_set = read_labels_set("./data/labels_set.json")
id2main = {labels_set['main2id'][k] : k for k in labels_set['main2id']}
id2sub = {labels_set['sub2id'][k] : k for k in labels_set['sub2id']}
print("labels read")

labels_num = len(id2main)
model_path = "./model_dir/100_epoches_original_ecco_bert/ecco_genre_main_ecco_bert_100_epoches.pt"

model = BertForSequenceClassification.from_pretrained("TurkuNLP/eccobert-base-cased-v1", num_labels=labels_num)
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/eccobert-base-cased-v1")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['net'])

print("model loaded")

import shap


from transformers import pipeline
model_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True) # -1 -> device CPU; >= 0 -> GPU
genre_list = [v for k, v in id2main.items()]
explainer = shap.Explainer(model_pipeline, output_names=genre_list)

pd_test_chunks = pd.read_csv("data/ecco_bert_seq_test_set_first_chunks.csv", sep="\t")

print("test chunks read")

def subwords_to_original(subword_text):
    out = ""
    for token in subword_text.split():
            if token in ["[CLS]", "[SEP]"]: continue
            if token.startswith("##"): 
                out += token.replace("##", "")
            else:
                out += " " + token
    return out.strip()
    
pd_test_chunks["text"] = pd_test_chunks["chunk_content"].apply(subwords_to_original)
chunk_texts = pd_test_chunks["text"].tolist()

# This is a kludge! I'll explain later...
chunk_texts2 = []
for text in chunk_texts:
    chunk_texts2.append(" ".join([word for word in text.split() if len(word)>2 and not word.isnumeric() and word.upper() not in ["THE", "AND"]]))

print("generating SHAP values...")
shap_values = explainer(chunk_texts2)

print("pickle dump")
import pickle
pickle_file = "data/ecco_bert_seq_test_shap_values.pickle"
pickle.dump(shap_values, open(pickle_file, 'wb'))

print("all done")
end = datetime.now()
print("ended:", end)
