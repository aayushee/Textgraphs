#!/usr/bin/env python3

import os
import warnings
from typing import List, Tuple, Iterable
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable, **kwargs) -> Iterable:
        return iterable

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

def get_hypothesis(df_q):
	delimiters="(A)","(B)","(C)","(D)","(E)","(1)","(2)","(3)","(4)"
	regexPattern = '|'.join(map(re.escape, delimiters))
	all_hyp=[]
	for id in range(len(df_q)) : 
		ques=df_q.loc[id,'question']
		qa=re.split(regexPattern, ques)
		ans=df_q.loc[id,'AnswerKey']
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
		fitb_q=get_fitb_from_question(qa[0])
		all_hyp.append(create_hypothesis(fitb_q,ch))
	df_q['qa']=all_hyp
	return df_q
	
def read_explanations(path: str) -> List[Tuple[str, str]]:
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nearest', type=int, default=100)
    parser.add_argument('tables')
    parser.add_argument('questions', type=argparse.FileType('r', encoding='UTF-8'))
    args = parser.parse_args()

    explanations = []

    for path, _, files in os.walk(args.tables):
        for file in files:
            explanations += read_explanations(os.path.join(path, file))

    if not explanations:
        warnings.warn('Empty explanations')

    df_q = pd.read_csv(args.questions, sep='\t', dtype=str)
    df_q = df_q[df_q['flags'].str.lower().isin(('success', 'ready'))].reset_index()
    #print('number of questions:',len(df_q))
    #print('length of explanations:',len(explanations))
    df_e = pd.DataFrame(explanations, columns=('uid', 'text'))
    df_q=get_hypothesis(df_q)
    new_df1=df_e.drop_duplicates(subset ="text", keep = 'first', inplace = False).reset_index(drop=True) 
    new_df=new_df1.drop_duplicates(subset ="uid", keep = 'first', inplace = False).reset_index(drop=True)
    new_df = new_df[new_df.uid != 'Good'].reset_index(drop=True)
    #print('length of unique explanations: ',len(new_df))
    vectorizer = TfidfVectorizer().fit(df_q['qa']).fit(new_df['text'])
    X_q = vectorizer.transform(df_q['qa'])
    #print(X_q.shape)
    X_e = vectorizer.transform(new_df['text'])
    #print(X_e.shape)
    X_dist = cosine_distances(X_q, X_e)
    for i_question, distances in tqdm(enumerate(X_dist), desc=args.questions.name, total=X_q.shape[0]):
        for i_explanation in np.argsort(distances)[:args.nearest]:
     #       print(i_explanation)
            #print('{}\t{}'.format(df_q.loc[i_question]['question'],df_e.loc[i_explanation]['text']))
            print('{}\t{}'.format(df_q.loc[i_question]['QuestionID'], new_df.loc[i_explanation]['uid']))



if '__main__' == __name__:
    main()