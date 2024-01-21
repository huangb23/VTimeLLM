#!/usr/bin/env python

from bert_score.scorer import BERTScorer


class BertScore:
    def __init__(self, lang="en", model_type="roberta-large"):
        self.lang = lang
        self.model_type = model_type
        self.bert = BERTScorer(model_type=model_type, lang=lang)

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        # convert dict to list of str
        cands = list(map(lambda x: x[0], res.values()))
        refs = list(map(lambda x: x[0], gts.values()))
        (P, R, F), hashname = self.bert.score(cands, refs, return_hash=True)
        #print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}')
        R = R.numpy()
        return R.mean(), R

    def method(self):
        return "BertScore"
