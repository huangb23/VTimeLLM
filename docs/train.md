# Training VTimeLLM
VTimeLLM adopts a three-stage training strategy. Please follow the instructions below to train VTimeLLM-7B model.


* Download [clip](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fcheckpoints&mode=list) and [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) weights, and place them into the 'checkpoints' directory.

* Download stage1 dataset from [this link](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json), and download stage2 and stage3 dataset from the [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fdata&mode=list). Place them into the 'data' directory.

```markdown
- VTimeLLM
    - checkpoints
        - clip
        	- ViT-L-14.pt
        - vicuna-7b-v1.5
        	- pytorch_model-00001-of-00002.bin
        	- ...
    - data
        - blip_laion_cc_sbu_558k.json
        - stage2.json
        - stage3.json
    - scripts
    	- stage1.sh
    	- stage2.sh
    	- stage3.sh
    	- ...
    - vtimellm
    - ...
```

If you want to train a Chinese version, you can download the [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) model and the translated Chinese [dataset](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fdata&mode=list).

* Download the pre-extracted features from the [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Ffeat&mode=list).

```shell
tar -xzvf stage1.tar.gz
cat stage2_part_* > stage2.tar.gz
tar -xzvf stage2.tar.gz
tar -xzvf stage3.tar.gz
```

* Train in three stages sequentially, and make sure to modify  '--feat_folder' in the script to the corresponding feature folder for each stage.

```shell
cd VTimeLLM
bash scripts/stage1.sh
bash scripts/stage2.sh
bash scripts/stage3.sh
```