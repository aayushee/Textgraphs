import pandas as pd
import re

# String used to indicate a blank
BLANK_STR = "___"



    
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

def read_label_data(labeled_file):
    q_dict={}
    with open(labeled_file,'r') as f1:
    
        for id, line in enumerate(f1):
            id_list=[]
            #print(id)
            text = line.strip().split('\t')
            if text[0] not in q_dict:
                id_list.append(text[1])
                q_dict[text[0]] = id_list
            else:
                new_list = q_dict[text[0]]
                new_list.append(text[1])
                q_dict[text[0]] = new_list
    return q_dict


mode='test'
labeled_file='questions/labeled_data_'+mode+'.tsv'
questions_file='questions/questions.'+mode+'.tsv'
facts_file='questions/explanations.tsv'
df_questions = pd.read_csv(questions_file, sep='\t').dropna(subset=["explanation"]).reset_index()
df_questions = df_questions[df_questions['flags'].str.lower().isin(('success', 'ready'))]
print(len(df_questions))
df_questions = remove_wrong_answer_choices(df_questions)
exp_dict=read_exp(facts_file)
data_file='questions/regression_'+mode+'.txt'
q_dict=read_label_data(labeled_file)
with open (data_file,'w') as wr:
    for i_q, question in df_questions.iterrows():
        true_exp=[]
        qid = question["QuestionID"]
        explanations = q_dict[qid]
        for exp_id in explanations:
            exp_text = exp_dict[exp_id]
            true_exp.append(exp_text)
        exp_string = ' . '.join(true_exp)
        wr.write(qid+'\t'+question['ProcessedQuestion']+'\t'+question['AnswerText']+'\t'+exp_string+'\n')