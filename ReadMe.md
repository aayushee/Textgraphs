All our experiments were done with python 3.6 Miniconda environment. 
Make sure following packages are available in the environment with pip install:
numpy, tqdm, scikit-learn, pandas, requests, urllib3, scipy, nltk, torch, torchvision.\
For BERT Baseline Ranking, the experiments were done on Google Colab GPU.

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
To train and evaluate:\
    python bert_reranker.py train\
To test:\
    python bert_reranker.py test\
If not training and testing the model, use the following link that has BERT trained model and test predictions:
https://drive.google.com/drive/folders/17nZaZ7t54CgGfCZ1rmi8NSoS3QFCAXnz?usp=sharing  \
model name: pytorch_model.bin\
prediction file on test data from model : tg2020_test_preds.npy\
test examples zipped file : examples_test_bert-base-uncased_140_tg2020.zip. unzip it & save in 'questions' folder.

2. Run write_predictions.py to write baseline ranking predictions as a text file [tg2020_test_predicted.txt]. It requires test predictions from BERT [tg2020_test_preds.npy] to be kept in the main directory
and test examples file [examples_test_bert-base-uncased_140_tg2020] in 'questions' directory.\
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
    python preprocess_ilp.py train\
    python preprocess_ilp.py test\
Run get_ilp_results.py to get scores from ILP for train and test data.\
    Set Solver Port and Solver Domain in lines 9 and 10 as set up in SemanticILP above.\
    Set n=3/4 to get 3 feature scores or 4 feature scores.\
    Set parameter mode=train/test 
    python get_ilp_results.py n mode
Run linear_regression.py to perform linear regression on the dataset followed by running regression_predictions.py to sort scores and regenerate ranked predictions.\
    Set n as 3/4 to get 3 or 4 features combined scores respectively.\
    python linear_regression.py n\
    Set n as 3/4 to get 3 or 4 features predictions respectively.\
    python regression_predictions.py n\
    The top 30 predictions from regression model created from above are also available in the 'predictions' folder:\
    regression_preds3.txt \
    regression_preds4.txt \
Run copy_predictions.py to add remaining predictions from BERT baseline after top-K and evaluate.py to evaluate the generated predictions.\
    Set n as 3/4 to get 3 or 4 features predictions respectively. \
    python copy_predictions.py n \
    generated files: pred_bert_regression_3scores_combined.txt/pred_bert_regression_4scores_combined.txt \
    python evaluate.py --gold questions/questions.test.tsv predictions/[generated file.txt]  

D) Summarizer Ranking
Run textrank.py file to get predictions on test data from summarizer.
 python preprocess_ilp.py test\
 python textrank.py\
 This will create the predictions file. To evaluate the predictions, run:\
 python evaluate.py --gold questions/questions.test.tsv predictions/predict_textrank.txt
 

