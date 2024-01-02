# Running VTimeLLM inference Offline
Please follow the instructions below to run the VTimeLLM inference on your local GPU machine. 

Note: Our demo requires approximately 18 GB of GPU memory.

### Clone the VTimeLLM repository

```shell
conda create --name=vtimellm python=3.10
conda activate vtimellm

git clone https://github.com/huangb23/VTimeLLM.git
cd VTimeLLM
pip install -r requirements.txt
```

### Download weights

* Download clip model and vtimellm model from the [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fcheckpoints&mode=list) and place them into the 'checkpoints' directory.
* Download [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) or [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b)  weights.

### Run the inference code


```shell
python -m vtimellm.inference --model_base <path to the Vicuna v1.5 weights> 
```

Alternatively, you can also choose to conduct multi-turn conversations in [Jupyter Notebook](inference.ipynb). Similarly, you need to set 'args.model_base' to the path of Vicuna v1.5.

If you want to run the VTimeLLM-ChatGLM version, please refer to the code in [inference_for_glm.ipynb](inference_for_glm.ipynb).

### Run the gradio demo

We have provided an offline gradio demo as follows:

```shell
cd vtimellm
python demo_gradio.py --model_base <path to the Vicuna v1.5 weights> 
```
