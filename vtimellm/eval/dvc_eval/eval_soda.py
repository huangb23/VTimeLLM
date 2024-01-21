import numpy as np
from .SODA.soda import SODA
from .SODA.dataset import ANETCaptions

def eval_tool(prediction, referneces=None, metric='Meteor', soda_type='c', verbose=False):

    args = type('args', (object,), {})()
    args.prediction = prediction
    args.references = referneces
    args.metric = metric
    args.soda_type = soda_type
    args.tious = [0.3, 0.5, 0.7, 0.9]
    args.verbose = verbose
    args.multi_reference = False

    data = ANETCaptions.from_load_files(args.references,
                                        args.prediction,
                                        multi_reference=args.multi_reference,
                                        verbose=args.verbose,
                                        )
    data.preprocess()
    if args.soda_type == 'a':
        tious = args.tious
    else:
        tious = None
    evaluator = SODA(data,
                     soda_type=args.soda_type,
                     tious=tious,
                     scorer=args.metric,
                     verbose=args.verbose
                     )
    result = evaluator.evaluate()

    return result

def eval_soda(p, ref_list,verbose=False):
    score_sum = []
    for ref in ref_list:
        r = eval_tool(prediction=p, referneces=[ref], verbose=verbose, soda_type='c')
        score_sum.append(r['Meteor'])
    soda_avg = np.mean(score_sum, axis=0) #[avg_pre, avg_rec, avg_f1]
    soda_c_avg = soda_avg[-1]
    results = {'soda_c': soda_c_avg}
    return results