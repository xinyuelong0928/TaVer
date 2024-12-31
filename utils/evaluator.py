import logging
import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js['idx']] = js['target']
    return answers

def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[int(idx)] = int(label)
    return predictions

def calculate_scores(answers, predictions):
    Acc = []
    y_trues, y_preds = [], []
    Result = []
    Fcount = 0
    Tcount = 0
    TTcount = 0
    TTTcount = 0
    FFcount = 0
    count = 0
    for key in answers:
        if key not in predictions:
            count = count + 1
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
        Acc.append(answers[key] == predictions[key])
        if answers[key] == 1:
            FFcount = FFcount + 1
        if answers[key] == 0:
            TTTcount = TTTcount + 1
        if answers[key] != predictions[key]:
            Fcount = Fcount + 1
            Result.append(key)
        if answers[key] == predictions[key]:
            Tcount = Tcount + 1
            if answers[key] == 0:
                TTcount = TTcount + 1
    scores = {}
    scores['Acc'] = np.mean(Acc)
    scores['Recall'] = recall_score(y_trues, y_preds, average="binary")
    scores['Prediction'] = precision_score(y_trues, y_preds)
    scores['F1'] = f1_score(y_trues, y_preds)
    return scores, Result, Fcount, Tcount, TTcount, TTTcount, FFcount, count, Acc

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument("--fusion_languages", type=str, default=None)
    parser.add_argument('--target_language', type=str, default=None)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions will be written.") 
    parser.add_argument('--answers', '-a',
                        help="filename of the labels, in txt format.")
    args = parser.parse_args()
    answers_file = args.answers
    language = args.target_language
    output_file = os.path.join(args.output_dir, "results_transfer_" + args.fusion_languages + "--"+language+".txt")
    answers = read_answers(os.path.join(answers_file, f"{language}_test.jsonl"))

    with open(output_file, "w") as result_file:
        predictions_file = os.path.join(args.output_dir,f"{args.fusion_languages}--{language}_predictions.txt")
        print(f"The prediction file has been generated to {output_file}.")
        if not os.path.exists(predictions_file):
            print(f"file not found.")
        predictions = read_predictions(predictions_file)
        scores, Result, Fcount, Tcount, TTcount, TTTcount, FFcount, count, ACC = calculate_scores(answers, predictions)
        result_file.write(json.dumps(scores) + "\n\n")

if __name__ == '__main__':
    main()