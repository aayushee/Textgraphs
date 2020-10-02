A) TF-IDF Ranking
Run baseline.py with required parameters to generate tf-idf ranking.

B) BERT Baseline Ranking
Run bert_reranker.py to generate training,eval and test examples and features from dataset.
Train model and test and save predictions.
Run write_predictions.py to write predictions as a text file.
run reorder_predictions.py to move redundant exp to the bottom and then evaluate the predicted file using evaluate.py.
This generated file is "predictions_bert.txt" available in the predictions folder.

Link to download BERT Trained Model:
https://drive.google.com/drive/folders/17nZaZ7t54CgGfCZ1rmi8NSoS3QFCAXnz?usp=sharing
model name: pytorch_model
predictions on test data from model : tg2020_test_preds.npy

C) ILP Rescoring and Regression Reranking
The code for ILP is hosted at: https://github.com/aayushee/semanticilp
It is in Scala and the instructions for setting it up are present in the readme file.

Run preprocess_ilp.py and process_regression_data.py to generate train and test data files for ILP model.
Run get_ilp_results.py to get scores from ILP for train and test data.
The parameters can be modified here to get 3 feature scores or 4 feature scores.
Run linear_regression.py to perform linear regression on the dataset followed by running regression_predictions.py to sort scores and regenerate ranked predictions.
Run copy_predictions.py to add remaining predictions from bert after top-K.
Run evaluate.py to evaluate the generated predictions.

The predictions from regression model are available in the 'predictions' folder.

