import argparse
import numpy as np
import torch
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import os

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def ensemble_preds(questions_file, facts_file, base_examples_file,
                   base_logits_file, pred_output_file):
    df_questions = pd.read_csv(questions_file, sep='\t')
    df_questions = df_questions[df_questions['flags'].str.lower().isin(('success', 'ready'))]

  #  df_facts = pd.read_csv(facts_file, sep='\t').drop_duplicates(subset=["uid"], keep="first").reset_index()

    base_examples = torch.load(base_examples_file)
    #base_logits = np.load(base_logits_file)
    print(base_examples[0].text_a,base_examples[0].text_b)
   # print("stop")    
    with open(base_logits_file,'rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    base_logits=out
    print('len of loaded logits: ',len(base_logits))
    base_logit_1 = base_logits[:, 1] - base_logits[:, 0]

    idx_start = 0
    prev_query = base_examples[0].text_a

    base_predictions = {}
    for i, example in enumerate(base_examples):
        if example.text_a == prev_query:
            continue

        relevant_logits = base_logit_1[idx_start:i]
        relevant_examples = base_examples[idx_start:i]
        sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                    reverse=True))
        qid = sorted_examples[0].guid.split('###')[0]
        #base_predictions[qid] = ['\t'.join(se.guid.split('###')) for se in sorted_examples]
        base_predictions[qid] = [se.text_b for se in sorted_examples]

        prev_query = example.text_a
        idx_start = i

    relevant_logits = base_logit_1[idx_start:]
    relevant_examples = base_examples[idx_start:]
    sorted_preds, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0],
                                                reverse=True))
    qid = sorted_examples[0].guid.split('###')[0]
   # base_predictions[qid] = ['\t'.join(se.guid.split('###')) for se in sorted_examples]
    base_predictions[qid] = [se.text_b for se in sorted_examples]

    print("len(base_predictions): {}".format(len(base_predictions)))
    print(type(base_predictions))
    print("Writing to file")
    with open(pred_output_file, "w") as f:
        for value in base_predictions.values():
            for elem in value:               
                f.write('{}\n'.format(elem))

    print("len(df_questions)={}".format(len(df_questions)))


def move_redundant_facts_to_end(questions_file, facts_file, fact_frequency_file, predictions_file,
                                output_predictions_file):
    df_questions = pd.read_csv(questions_file, sep='\t')
    df_facts = pd.read_csv(facts_file, sep='\t').drop_duplicates(subset=["uid"], keep="first").reset_index()
    fact_frequency = pickle.load(open(fact_frequency_file, "rb"))
    duplicate_facts = df_facts.loc[df_facts.duplicated(subset=["text"], keep=False)]
    for i, fact in duplicate_facts.iterrows():
        if fact_frequency.get(fact["uid"], int()) > 0:
            duplicate_facts = duplicate_facts.drop(index=[i])
    duplicate_facts = set(duplicate_facts["uid"])
    predictions = {}
    with open(predictions_file) as f:
        for line in f:
            ls = line.strip().split('\t')
            try:
                predictions[ls[0]].append(ls[1])
            except KeyError:
                predictions[ls[0]] = [ls[1]]

    new_predictions = {}
    for k, v in predictions.items():
        pred_start = []
        pred_end = []
        for fid in v:
            if fid in duplicate_facts:
                pred_end.append(fid)
            else:
                pred_start.append(fid)
        new_predictions[k] = '\n'.join([k + '\t' + p for p in pred_start + pred_end[::-1]])

    output_str = ''
    for qid in df_questions["questionID"]:
        if qid in new_predictions:
            print(qid)
            output_str += new_predictions[qid] + '\n'

    with open(output_predictions_file, "w") as f:
        f.write(output_str)


questions_file="questions/questions.test.tsv"
facts_file="questions/explanations.tsv"
base_examples_file="questions/examples_test_bert-base-uncased_140_tg2020"
base_logits_file="tg2020_test_preds.npy"
pred_output_file="predictions/tg2020_test_predicted_text.txt"

ensemble_preds(questions_file, facts_file, base_examples_file, base_logits_file, pred_output_file)
#        move_redundant_facts_to_end(args.questions_file, args.facts_file, args.fact_frequency_file,
 #                                   args.predictions_file, args.output_predictions_file)
