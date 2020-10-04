from collections import OrderedDict
from itertools import islice
import sys

path=""
number_of_features=sys.argv[1]
bert_fname=path+"predictions/predictions_bert.txt"
ilp_fname=path+"predictions/regression_preds"+str(number_of_features)+".txt"
new_fname=path+"predictions/pred_bert_regression_"+str(number_of_features)+"scores_combined.txt"

def write_pred(new_fname,pred_dict):
    print("writing predictions...")
    with open(new_fname,'w') as wr:
        for item in pred_dict.keys():
            explist=pred_dict[item]
            for exp in explist:
                wr.write(item+'\t'+exp+'\n')

def read_pred(bert_fname,ilp_fname):
    pred_dict=OrderedDict()
          
    print("reading ilp top 30 preds...")
    with open (ilp_fname,'r') as f1:
        for line in f1:
            li=[]
            text=line.strip().split('\t')
            if text[0] not in pred_dict:
                li.append(text[1])
                pred_dict[text[0]]=li
            else:
                new_li = pred_dict[text[0]]
                new_li.append(text[1])
                pred_dict[text[0]]=new_li
    print("reading remaining bert preds...")
    with open(bert_fname,'r') as f2:
        while True:
            lines = list(islice(f2, 30, 9726))
            if not lines:
                break
            else:
                for line in lines:
                    text=line.strip().split('\t')
                    new_li = pred_dict[text[0]]
                    new_li.append(text[1])
                    pred_dict[text[0]]=new_li
    return pred_dict
    

pred_dict = read_pred(bert_fname,ilp_fname)
write_pred(new_fname,pred_dict)