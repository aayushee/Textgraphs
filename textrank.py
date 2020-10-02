from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from operator import itemgetter 
import re
from itertools import islice


# String used to indicate a blank
BLANK_STR = "___"


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str) -> str:
    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    hypothesis = re.sub("__+", choice, fitb)
    return hypothesis


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?
        m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub("\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        #if 'Which of these' in fitb_question:
        #    return fitb_question.replace(BLANK_STR + " of these", BLANK_STR)
        return fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
    elif re.match(".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(" this[ \?]", " ___ ", question_str)

# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + BLANK_STR
    return fitb

def get_hypothesis(question,ch):
	
	fitb_q=get_fitb_from_question(question)
	hypothesis = create_hypothesis(fitb_q,ch)
	return hypothesis

def textrank(sentences, top_n, stopwords=None):
    """
    sentences = a list of sentences [[s1], [s2], ....]
    top_n = No.of sentences the summary should contain
    stopwords = a list of stopwords
    """
    #print('building matrix')
    S = build_similarity_matrix(sentences, stopwords) 
   # print(S)
    sentence_ranking = page_rank(S)
 
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranking), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:top_n])
    #print(selected_sentences)
    #summary = itemgetter(*selected_sentences)(sentences)
    return selected_sentences


def build_similarity_matrix(sentences, stop_words=None):
    S = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
               # print(sentences[i],sentences[j])
                S[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
               # print(S[i][j])
    #Normalize the matrix
    for i in range(len(S)):
        if S[i].sum()!=0.0:
            S[i] /= S[i].sum()
    
    return S

def page_rank(A, eps=0.0001, d=0.5):
    P = np.ones(len(A)) / len(A)
    while True:
        P_new = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((P_new - P).sum())
        if delta <= eps:
            return P_new
        P = P_new

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # Vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # Vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def read_exp(exp_file):
    exp_dict={}
    with open(exp_file,'r') as f1:
        for line in f1:
            text = line.strip().split('\t')
            if text[1] not in exp_dict:
                exp_dict[text[1]] = text[0]
    return exp_dict
    
 
def read_preds(fname,j,k): 
    with open(fname,'r') as br:
        lines = list(islice(br, j, k))
    return lines,k+1
    
def write_test_data(test_data):
    with open ('questions/textrank_test_data.txt','a') as wr2:
            wr2.write(test_data)
            
            
exp_file="questions/explanations.tsv"
exp_dict=read_exp(exp_file)   
stopwords = stopwords.words('english')
n = 17
sent_dict={}
fname='predictions/predictions_bert.txt'
count=0
with open ('predictions/predict_textrank.txt','w') as wr:
    with open ('questions/ilp_data_test.txt','r') as br:
        for i,line in enumerate(br):
            print(i)
            text = line.strip().split("\t")
            hypothesis = get_hypothesis(text[1],text[2]) 
            sents = text[3].split(' . ')
        #print(hypothesis)
            sent_dict[0]=hypothesis
            #j=count
            #k=j+9726
            #sents,count=read_preds(fname,j,k)
            
            for id,sent in enumerate(sents):
                sent_dict[id+1]=sent
            sentences=[]
            sentences.append(hypothesis.split())
            for sent in sents:
                words = sent.split()
                sentences.append(words)
        #print(sentences)

            summary_ids=textrank(sentences, n, stopwords)
            exp_text=[]
            for sent_id in summary_ids[1:]:
                exp_text.append(sent_dict[sent_id])
                #print(sent_dict[sent_id])
                exp_id=exp_dict[sent_dict[sent_id]]
                #wr.write(text[0]+'\t'+' '.join(sent)+'\n')
                wr.write(text[0]+'\t'+exp_id+'\n')
            exp_string=' . '.join(exp_text)
            test_data=text[0]+'\t'+text[1]+'\t'+text[2]+'\t'+exp_string+'\n'
            write_test_data(test_data)

