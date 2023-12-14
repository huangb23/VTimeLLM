Our dataset can be downloaded from [here](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fdata&mode=list). The data is organized in the following JSON format:

* source: The original sources of the dataset. This field is ['internvid'](https://huggingface.co/datasets/OpenGVLab/InternVid) in stage 2 and ['anet'](http://activity-net.org/download.html) or ['didemo'](https://drive.google.com/drive/u/0/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc) in stage 3.
* id: The ID of the video in the original dataset.
* conversations: The conversation data used for training. We utilize special tokens  (such as `<s0>`, `<e0>`) to indicate timestamps. These need to be replaced during training using information from the 'meta' field.
* meta:
  * split: refers to the extraction of a segment from the original  video as input, with two numbers denoting the start and end timestamps  of the extraction. If this field does not exist, the entire video is  used as input. 
  * duration: indicates the length of the input video.
  * token: represents the timestamp of special tokens appearing in  'conversations'.

Here is an example:

```json
{
    "source": "internvid",
    "id": "3n3oCNerzV0",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nI'd appreciate it if you could provide a detailed account of the events that occurred at different timestamps in the video."
        },
        {
            "from": "gpt",
            "value": "From <s0> to <e0>, woman is counting money with a pen on a white table. From <s1> to <e1>, two people shaking hands in front of a desk."
        }
    ],
    "meta": {
        "split": [
            129.8,
            184.0
        ],
        "duration": 54.2,
        "token": {
            "<s0>": 0,
            "<e0>": 9.4,
            "<s1>": 49.4,
            "<e1>": 53.2
        }
    }
}

```

Due to Internvid's data being sourced from YouTube, the 'id' field  provides the YouTube ID of the video. We download the [original video](https://www.youtube.com/watch?v=3n3oCNerzV0) and extract the segment from 129.8s to 184.0s as input.

If we sample $N=100$ frames as input, then `<s0>` should be replaced by $\frac{0}{54.2} * N = 00$,  `<e0>` should be replaced by $\frac{9.4}{54.2} * N = 17$. and `<s1>` and `<e1>` should be replaced in a similar manner. Finally, in this example, the question from 'human' remains unchanged,  while the 'answer' from 'gpt' is shown as below:

```
From 00 to 17, woman is counting money with a pen on a white table. From 91 to 98, two people shaking hands in front of a desk.
```



In the Stage 3 data, approximately 20k data originate from [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT). This subset of  data does not contain special tokens and the 'meta' field.
