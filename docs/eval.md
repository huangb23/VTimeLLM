# VTimeLLM-Vicuna Evaluation 

We provide evaluation code for VTimeLLM-Vicuna on dense video captioning and temporal video grounding tasks. Before proceeding, you should be able to run the inference code (refer to [offline_demo.md](offline_demo.md)). Below, we outline the evaluation process using the [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset as an example.

- Download the annotation JSON file for the dataset. For the ActivityNet Captions dataset, the test set annotation file is `val_2.json`. For other datasets, you need to preprocess them to match the JSON format of this dataset.
- Download the raw videos of the dataset and store them in a specific folder.
- (Optional) Pre-extract video features (refer to inference.py) and store them in a specific folder. (For the ActivityNet Captions dataset, we have extracted features for approximately 80% of the test set videos, placed in the [feat folder of the stage3 training](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Ffeat&mode=list). You can use them for incomplete testing.)
- Run the evaluation code, and record the model's responses in a log file. (Specify at least one of `feat_folder` and `video_folder`) :

```bash
python vtimellm/eval/eval.py \
     --data_path /path/to/val_2.json \
     --feat_folder /path/to/feat \
     --video_folder /path/to/video \
     --log_path /path/to/log \
     --model_base <path to the Vicuna v1.5 weights> 
```

* In order to compute metrics for the dense video captioning task, you need to install `pycocoevalcap` and `Java`. 

```bash
pip install pycocoevalcap
conda install conda-forge::openjdk
```

* Parse the log file and obtain the corresponding metrics.

```bash
python vtimellm/eval/metric.py \
     --data_path /path/to/val_2.json \
     --log_path /path/to/log
```