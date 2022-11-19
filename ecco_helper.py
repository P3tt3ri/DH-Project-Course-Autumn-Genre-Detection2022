import pandas as pd
import json

def read_labels_set(labels_set_path):
    with open(labels_set_path, "r") as f:
        labels2id = json.load(f)
    return labels2id

labels_set = read_labels_set("./data/labels_set.json")
id2main = {labels_set['main2id'][k] : k for k in labels_set['main2id']}
id2sub = {labels_set['sub2id'][k] : k for k in labels_set['sub2id']}
genres = [v for k, v in id2main.items()]

def subwords_to_original(subword_text):
    out = ""
    for token in subword_text.split():
            if token in ["[CLS]", "[SEP]"]: continue
            if token.startswith("##"): 
                out += token.replace("##", "")
            else:
                out += " " + token
    return out.strip()

df = pd.read_csv("data/ecco_bert_seq_test_set_first_chunks.csv", sep="\t")
df["text"] = df["chunk_content"].apply(subwords_to_original)    


def get_genres():
    return genres

def ecco_test_texts_as_list():
    return df["document_id"].tolist(), df["pred_id"].tolist(), df["pred_label"].tolist(), df["text"].tolist()

def doc_genre_probs(doc_id):
    return df.loc[df.document_id == doc_id, genres].values

