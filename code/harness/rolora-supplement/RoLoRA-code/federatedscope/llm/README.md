



Our implementation is adapted from FS-LLM. Here are steps of how to run the code.



### Step 1. Installation


```bash
# Create virtual environments with conda
conda create -n fs-llm python=3.9
conda activate fs-llm

# Install Pytorch>=1.13.0 (e.g., Pytorch==2.0.0)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install FS-LLM with editable mode
pip install -e .[llm]
```

Now, you have successfully installed the FS-LLM.



### Step 2. Data Preparation

```bash
cd sst2
python qnli2json.py
```
### Step 3. Run with config

Now, we can fine-tune a RoBERTa-Large on QNLI with RoLoRA.

```bash
python federatedscope/main.py --cfg federatedscope/llm/baseline/test_glue.yaml
```




