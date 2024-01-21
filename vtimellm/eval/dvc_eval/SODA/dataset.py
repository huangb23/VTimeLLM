import numpy as np
import json
from collections import defaultdict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .utils import iou, remove_nonascii


class ANETCaptions:
    def __init__(self, preds, gts, gt_vid, verbose=False):
        self.pred_keys = ['results']
        # self.pred_keys = ['results', 'version', 'external_data']
        self.verbose = verbose
        self.preds = preds
        self.gts = gts
        self.gt_vids = gt_vid
        self.tokenizer = PTBTokenizer()

    @classmethod
    def from_load_files(cls, gt_file, pred_file, multi_reference=True, verbose=False):
        gts, gt_vid = cls.load_ground_truth(gt_file, multi_reference=multi_reference, verbose=verbose)
        preds = cls.load_prediction(pred_file, verbose=verbose)
        # missing video
        gt_vid = [x for x in gt_vid if x in preds]
        gt_vid = cls.check_videos(gt_vid, preds.keys(),verbose=verbose)
        return cls(preds, gts, gt_vid, verbose=verbose)

    @classmethod
    def from_prediction(cls, gt_file, preds, multi_reference=True, verbose=False):
        results = {}
        for vid in preds['results']:
            results[vid] = sorted(preds["results"][vid], key=lambda x: x["timestamp"][0])
        gts, gt_vid = cls.load_ground_truth(gt_file, multi_reference=multi_reference)
        gt_vid = cls.check_videos(gt_vid, results.keys(),verbose=verbose)

        return cls(results, gts, gt_vid, verbose=verbose)
    
    @staticmethod
    def load_ground_truth(filenames, multi_reference=False, verbose=False):
        if verbose: 
            print(f"| Loading ground truths: {filenames}.")
        if isinstance(filenames, str):
            filenames = [filenames]
        gt_vids = set()
        gt = defaultdict(dict)
        gts = []
        for filename in filenames:
            if isinstance(filename, dict):
                _gt = filename
            else:
                with open(filename, "r") as f:
                    _gt = json.load(f) 
            gt_vids.update(_gt.keys())
            gts.append(_gt)
        if multi_reference is False:
            for vid in gt_vids:
                t, s = [], []
                for _g in gts:
                    if vid not in _g: 
                        continue
                    t += _g[vid]["timestamps"]
                    s += _g[vid]["sentences"]
                sort_t, sort_s = list(zip(*sorted(zip(t, s), key=lambda x: x[0][0])))
                gt[vid]["timestamps"] = sort_t
                gt[vid]["sentences"] = sort_s
            gts = [gt]
        if verbose:
            print(f"stats:\n\t n_files: {len(filenames)}, n_videos: {len(gt_vids)}")
        return gts, gt_vids 

    @staticmethod
    def load_prediction(filename, verbose=False):
        if verbose: print(f"\n| Loading predictions: {filename}.")
        if isinstance(filename, dict):
            pred = filename
        else:
            with open(filename, 'r') as f:
                pred = json.load(f)
        # If the json file doesn’t have enough attribute
        # if not all([key in pred.keys() for key in ["results"]]):
        #     raise IOError('Please input a correct format prediction file.')
        results = {}
        for vid in pred['results']:
            # if vid not in self.gt_vids: continue
            results[vid] = sorted(pred["results"][vid], key=lambda x: x["timestamp"][0])
        return results

    def preprocess(self):
        if self.verbose: print("\n| Preprocessing captions...")
        n_ref = len(self.gts)
        p_spliter = [0]
        g_spliter = [[0] for i in range(n_ref)]
        times = {}
        cur_preds = {}
        cur_gts = [{} for i in range(n_ref)]
        for i, vid in enumerate(self.gt_vids): 
            cur_preds.update({j+p_spliter[-1]:[{"caption": remove_nonascii(p["sentence"])}] for j,p in enumerate(self.preds[vid])})
            times[i] = [p["timestamp"] for p in self.preds[vid]]
            p_spliter.append(p_spliter[-1] + len(times[i]))
            for n in range(n_ref):
                if vid not in self.gts[n]: 
                    g_spliter[n].append(g_spliter[n][-1])
                    continue
                cur_gts[n].update({j+g_spliter[n][-1]:[{"caption": remove_nonascii(p)}] for j,p in enumerate(self.gts[n][vid]["sentences"])})
                g_spliter[n].append(g_spliter[n][-1] + len(self.gts[n][vid]["sentences"]))
        tokenize_preds = self.tokenizer.tokenize(cur_preds)
        tokenize_gts = [self.tokenizer.tokenize(j) for j in cur_gts]
        for i, vid in enumerate(self.gt_vids): 
            _p = [tokenize_preds[j] for j in range(p_spliter[i],p_spliter[i+1])]
            self.preds[vid] = {"timestamps":times[i], "sentences":_p}
            for n in range(n_ref):
                if vid not in self.gts[n]: continue
                _g = [tokenize_gts[n][j] for j in range(g_spliter[n][i],g_spliter[n][i+1])]
                self.gts[n][vid]["sentences"] = _g

    @staticmethod
    def check_videos(gold_vid, pred_vid, verbose=True):
        not_appear = set(gold_vid) - set(pred_vid)
        if len(not_appear) > 0 and verbose:
            print((f"Warning: some videos in ground truth file are not appeared in prediction file!\n"
                f"\t{len(not_appear)} videos are not predicted: {not_appear}"))
        return list(set(gold_vid) & set(pred_vid))

