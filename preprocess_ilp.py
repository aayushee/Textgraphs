from itertools import islice 
import pandas as pd
import re
from collections import OrderedDict 

def read_files(prediction_fname,exp_fname,test_fname):
    pred_dict=OrderedDict()
    exp_dict={}
    q_dict={}
    a_dict={}
    df_questions = pd.read_csv(test_fname, sep='\t')
    df_questions = df_questions[df_questions['flags'].str.lower().isin(('success', 'ready'))]
    for index, row in df_questions.iterrows(): 
        q_dict[row["QuestionID"]]= row["question"]
        a_dict[row["QuestionID"]]= row["AnswerKey"]
    with open(exp_fname,'r') as f1:
        for line in f1:
            text = line.strip().split('\t')
            exp_dict[text[0]] = text[1]
    
    with open(prediction_fname,'r') as f2:            
        for line in islice(f2, 30):  # get the first 30 lines
            pred_dict = process_line(line.strip(),pred_dict)
        while True:
            lines = list(islice(f2, 9697, 9727)) # get the top 30 for remaining qid 
            #lines = list(islice(f2, 9512, 9542)) #get top 30 from tf-idf
            if not lines:
                break
            else:
                for line in lines:
                    pred_dict = process_line(line.strip(),pred_dict)
            
    return pred_dict,exp_dict,q_dict,a_dict

def process_line(line,pred_dict):           
    text = line.split("\t")
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

def write_top30_pred(pred_dict,pred_top30_fname):
    print("writing top 30 prediction id file...")
    with open(pred_top30_fname,'w') as wr:
        for key in pred_dict.keys():
            values=pred_dict[key]           
            for val in values:
                wr.write(key+"\t"+val+"\n")
        

def write_top30_pred_text(pred_dict,pred_top30_text_fname,exp_dict):
    print("writing top 30 predictions text file...")
    with open(pred_top30_text_fname,'w') as wr:
        for key in pred_dict.keys():
            values=pred_dict[key]
            for val in values:
                wr.write(key+"\t"+exp_dict[val]+'\n')
 
def write_ilp_data(pred_dict, exp_dict, q_dict, ilp_fname):
    print("writing file for ilp...")
    delimiters="(A)","(B)","(C)","(D)","(E)","(1)","(2)","(3)","(4)"
    regexPattern = '|'.join(map(re.escape, delimiters))
    separator="//"
    with open (ilp_fname,'w') as wr:
        for key in pred_dict.keys():
            values = pred_dict[key]
            exp_texts=[]
            ques = q_dict[key]
            qa=re.split(regexPattern, ques)
            qtext=qa[0]
            options=separator.join(qa[1:])
            for val in values:
                exp_texts.append(exp_dict[val].strip())
            explanation = " . ".join(exp_texts)
            wr.write(key+"\t"+qtext+"\t"+options+"\t"+explanation+"\n")

def write_ilp_data_again(pred_dict, exp_dict, q_dict, ilp_fname, a_dict):
    print("writing file for ilp...")
    delimiters="(A)","(B)","(C)","(D)","(E)","(1)","(2)","(3)","(4)"
    regexPattern = '|'.join(map(re.escape, delimiters))
    separator="//"
    with open (ilp_fname,'w') as wr:
        for key in pred_dict.keys():
            values = pred_dict[key]
            exp_texts=[]
            ques = q_dict[key]
            qa=re.split(regexPattern, ques)
            qtext=qa[0]
            ans=a_dict[key]
            if ans=="A" or ans=="1":
                ch=qa[1]
            elif ans=="B" or ans=="2":
                ch=qa[2]
            elif ans=="C" or ans=="3":
                ch=qa[3]
            elif ans=="D" or ans=="4":
                ch=qa[4]
            else:
                ch=qa[5]
            for val in values:
                exp_texts.append(exp_dict[val].strip())
            explanation = " . ".join(exp_texts)
            wr.write(key+"\t"+qtext+"\t"+ch+"\t"+explanation+"\n")

model_name='bert'            
mode='test'
prediction_fname = 'predictions/predictions_'+model_name+'.txt'
exp_fname = 'questions/explanations.tsv'
pred_top30_fname = 'predictions/'+model_name+'_top30.txt'
pred_top30_text_fname = 'predictions/'+model_name+'_top30_text.txt'
test_fname = 'questions/questions.'+mode+'.tsv'
ilp_fname = 'questions/ilp_data_'+mode+'.txt'
pred_dict, exp_dict, q_dict, a_dict = read_files(prediction_fname,exp_fname,test_fname)
print("questions and predictions processed: ",len(pred_dict))
write_top30_pred(pred_dict,pred_top30_fname)
write_top30_pred_text(pred_dict,pred_top30_text_fname,exp_dict)
#write_ilp_data(pred_dict, exp_dict, q_dict, ilp_fname)
write_ilp_data_again(pred_dict, exp_dict, q_dict, ilp_fname, a_dict)