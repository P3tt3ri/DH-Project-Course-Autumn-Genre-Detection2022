# DH-Project-Course-Autumn-Genre-Detection2022
Repository for files of the genre detection group for the Digital Humanities Project Course II (autumn 2022) of University of Helsinki

Files in the root folder:

- **ECCO_explanation_nmf.ipynb**: ECCO package (https://github.com/jalammar/ecco; not to be confused with ECCO collection) demo. 
- **ecco_helper.py**: Module containing some utility functions
- **LIME_demo.ipynb**: Jupyter notebook demonstrating the use of **lime** package (note: visualizations generated by **show_in_notebook** method don't get saved in the .ipynb file so those can be examined only by running the notebook) 
- **SHAP_demo.ipynb**: SHAP package demo notebook
- **SHAP_batch.py** and **SHAP_batch_run.sh**: scripts for generating SHAP values for the whole test set on Puhti. 
- **transformers_interpret_demo.ipynb**: transformers_interprets package demo.

Subfolders:

- **data**: json and csv files containing ECCO data
  - **labels.json**:                               ECCO genre/subgenre labels and their numeric IDs
  - **test.csv**:                                  Document IDs of the test set documents
  - **ecco_bert_seq_test_set_first_chunks.csv**:   The first text chunks of the test set: doc id, predicted genre, the tokenized text of the chunk, predicted probabilities for each genre  

- **model_dir**: ECCO-BERT PyTorch model files (just the one at the moment)
  - **ecco_genre_main_ecco_bert_100_epoches.pt**:  The fine tuned ECCO-BERT-seq model for main genre prediction. **Note:** The file was too large to push directly into the repository, so we created a release (2.0) that includes it. Just copy the file into the **model_dir** folder after downloading the release and unzipping the source files.
  
