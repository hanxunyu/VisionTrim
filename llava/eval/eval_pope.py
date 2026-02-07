# -- coding: utf-8 --
import os
import json
import argparse
total = 0
correct = 0

def eval_pope(answers, label_file):
    global total, correct
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    total += (TP + TN + FP + FN)
    correct += (TP + TN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    
    answers = []
    with open(args.result_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    answer = json.loads(line)
                    answers.append(answer)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {line}")
                    print(f"Error message: {e}")

    if not answers:
        print("Warning: No valid JSON data read")

    total = 0
    correct = 0
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")
    print(total, correct, correct / total)
