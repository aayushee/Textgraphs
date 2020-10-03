import urllib
import urllib.request
import urllib.parse
import json
from collections import OrderedDict 
import pandas as pd
import sys

SOLVER_DOMAIN = '192.168.1.103'
SOLVER_PORT = '9003'
 
mode=sys.argv[1]
irange=sys.argv[2]
if int(irange)==3:
    colname="QID,EID,QPA,PAA,IPA"
else:
    colname="QID,EID,QPA,PAA,IPA,ISA"

result_fname = 'questions/ilp_'+str(irange)+'scores_'+mode+'.csv'
with open (result_fname,'w') as wr:
    wr.write(colname+"\n")

def write_result(qid,exp,qpa,paa,ipa,isa):
    
    with open (result_fname,'a') as wr:
        for i in range (0,len(exp)):
            if len(isa)<1:
                wr.write(qid+','+exp[i]+','+qpa[i]+','+paa[i]+','+ipa[i]+'\n')
            else:
                wr.write(qid+','+exp[i]+','+qpa[i]+','+paa[i]+','+ipa[i]+','+isa[i]+'\n')

    
def read_exp(exp_file):
    exp_dict={}
    with open(exp_file,'r') as f1:
        for line in f1:
            text = line.strip().split('\t')
            if text[1] not in exp_dict:
                exp_dict[text[1].strip()] = text[0].strip()
    return exp_dict

def read_data(fname,exp_dict,irange):
    #result = OrderedDict() 
    j=0
    
    qids=[]
    with open (fname,'r') as fr:
        for i,line in enumerate(fr):
            print(i)
            text = line.split('\t')
            QUESTION = text[1].strip()
            ANSWERS = text[2]
            SNIPPET = text[3].strip()
            #q1 = urllib.parse.quote(QUESTION)
            #q2 = urllib.parse.quote(ANSWERS)
            #q3 = urllib.parse.quote(SNIPPET)
            data = {"question":QUESTION,"options":ANSWERS,"snippet":SNIPPET}
            data = urllib.parse.urlencode(data)
            
            url = 'http://'+SOLVER_DOMAIN+':'+SOLVER_PORT+'/solveQuestion?'+data
            req = urllib.request.Request(url)
            #response = None
            attempts = 0
            indexes=[]
            qpa=[]
            paa=[]
            ipa=[]
            isa=[]
            #while response is None:
            while attempts < 1:
                try:
                    if len(url)<4000:
                        response = urllib.request.urlopen(req,timeout=700)
                        out = response.read().decode("utf-8")
                        data = json.loads(out) 
                        data1 = data['log'].replace('List(','')           
                        data2 = data1.replace(')','')
                        data3 = data2.replace('(','')
                        indexes = data3.split(',')
                        break
                    else:
                        break
                    
                except BaseException as error:
                    print('An exception occurred: {}'.format(error))
                    attempts+=1
            
            sents = SNIPPET.split(' . ')
           # print(sents)
            exp_ids=[]
            if len(indexes) >1:
                for i in range(0,len(indexes),int(irange)):
                
                    qpa.append(indexes[i].strip())
                    paa.append(indexes[i+1].strip())
                    ipa.append(indexes[i+2].strip())
                    if int(irange)==4:
                        isa.append(indexes[i+3].strip())
                for sent in sents:
                    exp_ids.append(exp_dict[sent.strip()])
                write_result(text[0].strip(),exp_ids,qpa,paa,ipa,isa)
                qids.append(text[0].strip())
            else:
                j+=1
                print('soln infeasible')
             
    print("questions for which solution infeasible: ",j)
    return qids

def read_labels(fname):
    label_dict={}
    with open (fname,'r') as rr:
        for line in rr:
            text=line.strip().split('\t')
            if text[0] not in label_dict:
                label_list = []
                label_list.append(text[2].strip())
                label_dict[text[0]] = label_list
            else:
                label_list = label_dict[text[0]]
                label_list.append(text[2].strip())
                label_dict[text[0]] = label_list 
    return label_dict

def write_labeled_result(label_dict,result_fname,qids):
    labels_list=[]
    for id in qids:
        labels_list.extend(label_dict[id])
    df_result = pd.read_csv(result_fname)
    df_result['Label']=labels_list
    df_result.to_csv(result_fname)
    
exp_file = 'questions/explanations.tsv'
exp_dict=read_exp(exp_file)
fname = 'questions/ilp_data_'+mode+'.txt'
qids=read_data(fname,exp_dict,irange)
label_fname="questions/labeled_data_"+mode+".tsv"
if mode=='train':
    label_dict=read_labels(label_fname)
    write_labeled_result(label_dict,result_fname,qids)
