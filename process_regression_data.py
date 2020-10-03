from collections import OrderedDict
import re
import pandas as pd
import random

def read_pred(prediction_fname):
    pred_dict=OrderedDict()
    with open(prediction_fname,'r') as f2:

        for id, line in enumerate(f2):
            id_list=[]
            #print(id)
            text = line.strip().split('\t')
            if text[0] not in pred_dict:
                id_list.append(text[1])
                pred_dict[text[0]] = id_list
            else:
                new_list = pred_dict[text[0]]
                new_list.append(text[1])
                pred_dict[text[0]] = new_list
    return pred_dict

def remove_wrong_answer_choices(df):
    anslist=[]
    qlist=[]
    for id,question in df.iterrows():
        delimiters="(A)","(B)","(C)","(D)","(E)","(1)","(2)","(3)","(4)"
        regexPattern = '|'.join(map(re.escape, delimiters))
        ques=question['question']
        qa=re.split(regexPattern, ques)
        ans=question['AnswerKey']
        #print(qa)
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
        anslist.append(ch)
        qlist.append(qa[0])
    df['ProcessedQuestion']=qlist
    df['AnswerText']=anslist
    return df

def read_exp(exp_file):
    exp_dict={}
    with open(exp_file,'r') as f1:
        for line in f1:
            text = line.strip().split('\t')
            exp_dict[text[0]] = text[1]
    return exp_dict

path=""
facts_file=path+"questions/explanations.tsv"
mode = 'train'
if mode=='train' or mode=='dev':
    prediction_fname = path+'tfidf_'+mode+'_top30.txt'
else:
    prediction_fname = path + 'predictions/bert_top30.txt'
questions_file = path+'questions/questions.'+mode+'.tsv'
df_questions = pd.read_csv(questions_file, sep='\t').dropna(subset=["explanation"]).reset_index()
df_questions = df_questions[df_questions['flags'].str.lower().isin(('success', 'ready'))]
#df_questions = remove_wrong_answer_choices(df_questions)
pred_dict=read_pred(prediction_fname)
exp_dict=read_exp(facts_file)
data_file=path+'questions/labeled_data_'+mode+'.tsv'
with open (data_file,'w') as wr:
    for i_q, question in df_questions.iterrows():
        qid = question["QuestionID"]
        print(qid)
        explanations = [e.split('|')[0] for e in question["explanation"].split(' ')]
        for exp_id in explanations:
            wr.write(qid+'\t'+exp_id+'\t1\n')
        preds = pred_dict[qid]
        #print(preds)
        #print(explanations)
        pred_copy = [i for i in preds if i not in explanations]
        #print(pred_copy)
        reqd_preds = 30-len(explanations)
        #print(len(explanations),reqd_preds,len(pred_copy))
        wrong_preds = random.sample(pred_copy,reqd_preds)
        for pred in wrong_preds:
            wr.write(qid+'\t'+pred+'\t0\n')
