All our experiments were done with python 3.5 Miniconda environment. 
For BERT model, the experiments were done on Google Colab GPU.

A) TF-IDF Ranking\
Run baseline.py with the following parameters to generate tf-idf baseline ranking:\
        python baseline.py -n 30 tablestore/v2.1/tables questions/questions.test.tsv>predictions/tfidf_top30.txt\
To evaluate it, run:\
        python evaluate.py --gold questions/questions.test.tsv predictions/tfidf_top30.txt\
        This gives MAP on TF-IDF baseline ranking.

B) BERT Baseline Ranking
1. Run bert_reranker.py to generate training,eval and test examples and features from dataset. Then train and test model and save test predictions as follows:\
Requirements:\
pip install transformers==2.2.1\
pip install tensorboardX\
pip install pandas\
install apex:\
%%writefile setup.sh\
git clone https://github.com/NVIDIA/apex  \
cd apex\
pip install -v --no-cache-dir ./    \
  sh setup.sh\
To train:\
    python bert_reranker.py\
To test:\
    Replace line 887 with:     args = Args2()\
    python bert_reranker.py\
If not training and testing the model, use the following link that has BERT trained model and test predictions:
https://drive.google.com/drive/folders/17nZaZ7t54CgGfCZ1rmi8NSoS3QFCAXnz?usp=sharing  \
model name: pytorch_model.bin\
prediction file on test data from model : tg2020_test_preds.npy\
test examples zipped file : examples_test_bert-base-uncased_140_tg2020.zip. unzip it & save in 'questions' folder.

2. Run write_predictions.py to write baseline ranking predictions as a text file [tg2020_test_predicted.txt]. It requires test predictions from BERT [tg2020_test_preds.npy]
and test examples file [examples_test_bert-base-uncased_140_tg2020].\
    python write_predictions.py 

3. Run reorder_predictions.py to move duplicate predictions to the bottom and then evaluate the predicted file using evaluate.py:\
    python reorder_predictions.py\
    python evaluate.py --gold questions/questions.test.tsv predictions/predictions_bert.txt  \
    This gives MAP on BERT baseline ranking.

C) ILP Rescoring and Regression Reranking\
The code for ILP is hosted at: https://github.com/aayushee/semanticilp   \
It is in Scala. Clone the repository, further instructions for setting it up are present in the readme file in the repository.

Run baseline.py and process_regression_data.py to generate labeled file for train data :\
    python baseline.py -n 30 tablestore/v2.1/tables questions/questions.train.tsv>predictions/tfidf_train_top30.txt  \
    python process_regression_data.py\
Run preprocess_ilp.py to generate train and test data file for ILP model.\
    Set mode==train/test in line 115.\
    python preprocess_ilp.py\
Run get_ilp_results.py to get scores from ILP for train and test data.\
    Set Solver Port and Solver Domain in lines 8 and 9 as set up in SemanticILP above.\
    Set parameter 'irange' can be modified on line 12 to get 3 feature scores or 4 feature scores.\
    Set parameter mode==train/test on line 11.\
    python get_ilp_results.py\
Run linear_regression.py to perform linear regression on the dataset followed by running regression_predictions.py to sort scores and regenerate ranked predictions.\
    Set number_of_features in line 9 as 3/4 to get 3 or 4 features combined scores respectively.\
    python linear_regression.py\
    Set number_of_features in line 7 as 3/4 to get 3 or 4 features predictions respectively.\
    python regression_predictions.py\
    The top 30 predictions from regression model created from above are also available in the 'predictions' folder:\
    regression_preds2.txt \
    regression_preds3.txt \
    regression_preds4.txt \
Run copy_predictions.py to add remaining predictions from bert after top-K and evaluate.py to evaluate the generated predictions.\
    Set number_of_features in line 5 as 3/4 to get 3 or 4 features predictions respectively. \
    python copy_predictions.py \
    generated files: pred_bert_regression_3scores_combined.txt/pred_bert_regression_4scores_combined.txt \
    python evaluate.py --gold questions/questions.test.tsv predictions/[generated file.txt]  



