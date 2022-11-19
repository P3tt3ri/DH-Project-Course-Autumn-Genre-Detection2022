# DH-Project-Course-Autumn-Genre-Detection2022
Repository for files of the genre detection group for the Digital Humanities Project Course II (autumn 2022) of University of Helsinki

Files in the root folder:

- **ecco_helper.py**: Module containing some utility functions
- **LIME_demo.ipynb**: Jupyter notebook demonstrating the use of **lime** package

Subfolders:
- data: json and csv files containing ECCO data
  - labels.json                               ECCO genre/subgenre labels and their numeric IDs
  - test.csv                                  Document IDs of the test set documents
  - ecco_bert_seq_test_set_first_chunks.csv   The first text chunks of the test set: doc id, predicted genre, the tokenized text of the chunk, predicted probabilities for each genre  
- model_dir:
  - ecco_genre_main_ecco_bert_100_epoches.pt  The fine tuned ECCO-BERT-seq model for main genre prediction.  
