import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dvc_eval import eval_dvc, eval_soda

import json
import argparse
import re
import difflib

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")


def merge_similar_sentences(data):
    if not data: return data
    merged_data = []
    current_sentence = data[0]["sentence"]
    current_timestamp = data[0]["timestamp"]
    for i in range(1, len(data)):
        next_sentence = data[i]["sentence"]
        next_timestamp = data[i]["timestamp"]
        if difflib.SequenceMatcher(None, current_sentence, next_sentence).ratio() > 0.98 and -1 <= next_timestamp[0] - current_timestamp[1] <= 1:
            current_timestamp = [current_timestamp[0], next_timestamp[1]]
        else:
            merged_data.append({"sentence": current_sentence, "timestamp": current_timestamp})
            current_sentence = next_sentence
            current_timestamp = next_timestamp
    merged_data.append({"sentence": current_sentence, "timestamp": current_timestamp})
    return merged_data

def captioning_metrics(all_logs, data_path):
    logs = [x for x in all_logs if x['task'] == 'captioning']
    pred = {}
    for log in logs:
        id = log['video_id']
        answer = log['answer']
        pred[id] = []
        try:
            items = json.loads(re.search(r'\[.*\]', answer).group(0))
            for item in items:
                pred[id].append({
                        'timestamp': [int(item['timestamps'][5:7]), int(item['timestamps'][-2:])],
                        'sentence': item['event'],
                    })
        except Exception as e:
            print("Error", e, answer)
        

    gt_js = json.load(open(data_path))
    gt_js = {k: v for k, v in gt_js.items() if k in pred.keys()}

    
    for id, items in list(pred.items()): 
        items = merge_similar_sentences(items)
        duration = gt_js[id]['duration']
        for item in items:
            item['timestamp'][0] = item['timestamp'][0] * duration / 100
            item['timestamp'][1] = (item['timestamp'][1] + 1) * duration / 100
        pred[id] = items
     
    pred_result = {'results': pred}

    metrics = eval_soda(pred_result, [gt_js])
    metrics.update(eval_dvc(pred_result, [gt_js], 
                tious=[0.3, 0.5, 0.7, 0.9], 
                distances=[],
                max_proposals_per_video=1000, 
                verbose=False, 
                no_lang_eval=False))
    print(f"Found {len(pred)} logs")
    metrics = {k: v * 100 for k, v in metrics.items() if k in ['soda_c', 'METEOR', 'CIDEr']}
    return metrics


def grounding_metrics(all_logs):
    ious = [x['info']['iou'] for x in all_logs if x['task'] == 'grounding']
    l = len(ious)
    print(f"Found {l} logs")
    if l == 0: return
    metrics = {
        "mIoU": sum(ious) / l * 100
    }
    for m in [0.3, 0.5, 0.7]:
        metrics[f"R1@{m}"] = sum(iou >= m for iou in ious) / l * 100
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/example_log.txt')
    parser.add_argument("--task", type=str, default='all', choices=['all', 'grounding', 'captioning'])
    parser.add_argument("--data_path", type=str, default='vtimellm/eval/data_example.json')
    args = parser.parse_args()

    logs = []
    with open(args.log_path) as f:
        for line in f:
            try:
                json_data = json.loads(line)
                logs.append(json_data)
            except Exception as e:
                print(e, line)

    if args.task in ['captioning', 'all']:
        print("====================== Captioning =====================")
        print_metrics(captioning_metrics(logs, args.data_path))
    if args.task in ['grounding', 'all']:
        print("====================== Grounding ======================")
        print_metrics(grounding_metrics(logs))
