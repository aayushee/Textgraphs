from collections import OrderedDict
from itertools import islice


def read_exp(exp_file):
    exp_dict={}
    duplicates=[]
    with open(exp_file,'r') as f1:
        for line in f1:
            text = line.strip().split('\t')
            if text[1] not in exp_dict:
                exp_dict[text[1]] = text[0]
            else:
                duplicates.append(text[0])
    return duplicates

def read_pred(prediction_fname,duplicates):
    pred_dict=OrderedDict()
    with open(prediction_fname,'r') as f2:

        for id,line in enumerate(f2):
            id_list=[]
            if (id%9727==0):
                print(id)
            text=line.strip().split('\t')
            if text[0] not in pred_dict:
                id_list.append(text[1])
                pred_dict[text[0]] = id_list
            else:
                new_list=pred_dict[text[0]]
                new_list.append(text[1])
                pred_dict[text[0]]=new_list

    for item in pred_dict.keys():
        again_list=pred_dict[item]
      #  print(item)
        list_copy=again_list[:]
        for id in again_list:
            if id in duplicates:
                list_copy.remove(id)
                list_copy.append(id)
       # print(list_copy)
        pred_dict[item]=list_copy
    return pred_dict



def write_pred(new_fname,pred_dict):
    print("writing predictions...")
    with open(new_fname,'w') as wr:
        for item in pred_dict.keys():
            explist=pred_dict[item]
            for exp in explist:
                wr.write(item+'\t'+exp+'\n')


path="D:/Worldtree/"
questions_file=path+"questions/questions.test.tsv"
facts_file=path+"questions/explanations.tsv"
prediction_fname=path+"predictions/tg2020_test_predicted.txt"
#prediction_fname=path+"predictions/small_pred.txt"
new_fname=path+"predictions/predictions_bert.txt"
duplicates=read_exp(facts_file)
print(len(duplicates))
#print(duplicates)
pred_dict=read_pred(prediction_fname,duplicates)
write_pred(new_fname,pred_dict)

