import pandas as pd
from collections import OrderedDict 


mode='test'
df_test = pd.read_csv('questions/ilp_threescores_'+mode+'.csv')

from collections import defaultdict
result=defaultdict(dict)
qids=[]
for i,row in df_test.iterrows():
    qid=row['QID']
    result[qid][row['EID']]=row['CombinedScoreScaled']
print("predictions from ilp and regression: ",len(result))

def process_line(line,pred_dict):           
    text = line.strip().split("\t")
    q_id = text[0]
    exp_id = text[1]
    if q_id not in pred_dict:
        exp_list = []
        exp_list.append(exp_id)
        pred_dict[q_id] = exp_list
        return pred_dict
    else:
        exp_list = pred_dict[q_id]
        exp_list.append(exp_id)
        pred_dict[q_id] = exp_list  
        return pred_dict

pred_dict=OrderedDict()
with open('predictions/bert_top30.txt','r') as rb:
    for line in rb:
        pred_dict=process_line(line,pred_dict)
    
with open('predictions/regression_preds3.txt','w') as wr:
    for key in pred_dict.keys():
        if key in result:
            expscores=result[key]     
            sort_orders = sorted(expscores.items(), key=lambda x: x[1], reverse=True)
            for i in sort_orders:
                wr.write(key+"\t"+i[0]+"\n")
        else:
            expids=pred_dict[key]
            for exp in expids:
                wr.write(key+"\t"+exp+"\n")
