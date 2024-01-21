import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.append(root_dir)

import clip
import re
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import VideoExtractor
from vtimellm.inference import inference

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--task", type=str, default='all', choices=['all', 'grounding', 'captioning'])
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/example_log.txt')
    args = parser.parse_args()
    return args

def iou(outputs, gt):
    matches = re.search(r"(\d{2}) (to|and) (\d{2})", outputs)
    if not matches:
        return 0
    from_number = float(matches.group(1)) / 100
    to_number = float(matches.group(3)) / 100
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)


def write_log(log_path, video_id, task, query_id, answer, info=None):
    log = {
        'video_id': video_id,
        'task': task,
        'query_id': query_id,
        'answer': answer
    }
    if info is not None:
        log['info'] = info
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

questions = {
    'grounding': ['During which frames can we see {}?'],
    'captioning': ['Could you please describe the events in the video in detail? Be specific about the activities of individuals, their surroundings, and interactions with others. The output should be in JSON format, structured as follows: {"event": "xx", "timestamps": "from xx to xx"}.']
}

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)

    if args.video_folder is not None:
        clip_model, _ = clip.load(args.clip_path)
        clip_model.eval()
        clip_model = clip_model.cuda()

        video_loader = VideoExtractor(N=100)

        transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    js = json.load(open(args.data_path))
    for id, data in tqdm(js.items()):
        features = None

        if args.feat_folder is not None:
            feat_path = os.path.join(args.feat_folder, f"{id}.npy")
            if os.path.isfile(feat_path):
                features = torch.from_numpy(np.load(feat_path)).cuda()

        if features is None and args.video_folder is not None:
            for ext in ['mp4', 'mkv', 'webm']:
                video_path = os.path.join(args.video_folder, f"{id}.{ext}")
                if os.path.isfile(video_path):
                    _, images = video_loader.extract({'id': None, 'video': video_path})

                    images = transform(images / 255.0)
                    images = images.to(torch.float16)
                    with torch.no_grad():
                        features = clip_model.encode_image(images.to('cuda'))

        if features is None:
            print(f'Can not find video {id}')
            continue
 
        if args.task in ['captioning', 'all']:
            for query_id, query in enumerate(questions['captioning']):
                answer = inference(model, features, "<video>\n " + query, tokenizer)
                write_log(args.log_path, id, 'captioning', query_id, answer)
      
        if args.task in ['grounding', 'all']:
            for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]

                for query_id, query in enumerate(questions['grounding']):
                    answer = inference(model, features, "<video>\n" + query.format(sentence), tokenizer)
                    gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
                    u = iou(answer, gt)
                    write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
