# SODA
This repository is the imprimentation of "SODA: Story Oriented Dense Video Captioning Evaluation Flamework" published at ECCV 2020 [pdf](https://fujiso.github.io/publications/ECCV2020_soda.pdf).
SODA measures the performance of video story description systems.

## Update
v1.1 (2021/5)
* Added new option "--multi_reference" to deal with multiple reference.  
  SODA selects the reference that has the maximum f1 for each video, and returns macro averaged scores.  
* Fixed BertScore import

## Requirements
python 3.6+ (developed with 3.7)
* Numpy
* tqdm
* [pycocoevalcap (Python3 version)](https://github.com/salaniz/pycocoevalcap)
* BERTScore (optional)

## Usage
You can run SODA by specifying the path of system output and that of ground truth.
Both files should be the json format for ActivityNet Captions.
```bash
python soda.py -s path/to/submission.json -r path/to/ground_truth.json 
```

You can run on the multiple reference setting, with `--multi_reference` option.
```bash
python soda.py --multi_reference -s path/to/submission.json -r path/to/ground_truth1.json path/to/ground_truth2.json
```

You can try other sentence evaluation metrics, e.g. CIDEr and BERTScore, with `-m` option.
```bash
python soda.py -s path/to/submission.json -m BERTScore
```

## Sample input file
Please use the same format as [ActivityNet Challenge](http://activity-net.org/index.html)
```
{
  version: "VERSION 1.0",
  results: {
    "sample_id" : [
        {
        sentence: "This is a sample caption.",
        timestamp: [1.23, 4.56]
        },
        {
        sentence: "This is a sample caption 2.",
        timestamp: [7.89, 19.87]
        }
    ]
  }
  external_data: {
    used: False,
    }
}
```

## Reference
```
@inproceedings{Fujita2020soda,
  title={SODA: Story Oriented Dense Video Captioning Evaluation Flamework},
  author={Soichiro Fujita and Tsutomu Hirao and Hidetaka Kamigaito and Manabu Okumura and Masaaki Nagata},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  month={August},
  year={2020},
}
```

## LICENSE
NTT License

According to the license, it is not allowed to create pull requests.
Please feel free to send issues.
