#!/usr/bin/env python
import numpy as np
#from moverscore_v2 import get_idf_dict, word_mover_score 
from moverscore import get_idf_dict, word_mover_score 
from collections import defaultdict

class MoverScore:
    def __init__(self, lang="en", model_type=None):
        self.lang = lang
        self.model_type=model_type
        #self.model = load_model(model_type=model_type, lang=lang)
        self.idf_dict_ref = None
        self.idf_dict_hyp = None

    def compute_score(self, gts, res):
        assert gts.keys()==res.keys()
        assert self.idf_dict_hyp is not None and self.idf_dict_hyp is not None
        # convert dict to list of str
        cands = list(map(lambda x:x[0], res.values()))
        refs = list(map(lambda x:x[0], gts.values()))

        scores = word_mover_score(refs, cands, self.idf_dict_ref, self.idf_dict_hyp, \
                                          stop_words=[], n_gram=1, remove_subwords=True)
        #print(np.mean(scores), max(scores))
        return np.mean(scores), scores

    def make_dict(self, all_gts, all_res, vids):
        gold = []
        pred = []
        for vid in vids:
            gold.extend(all_gts[vid]["sentences"])
            pred.extend([pred["sentence"] for pred in all_res[vid]])
        self.idf_dict_ref = get_idf_dict(gold)
        self.idf_dict_hyp = get_idf_dict(pred)
        #print(self.idf_dict_hyp)

    def method(self):
        return "MoverScore"
