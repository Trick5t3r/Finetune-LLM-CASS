# Fine-tuning a Small Language Model (SLM) for Summarization

## 📌 Project Overview
This project aims to fine-tune a **Small Language Model (SLM)** (e.g., Qwen2.5-0.5B-Instruct, Atlas-Chat-2B, mT5-Base) on a **summarization task** using a **non-English dataset**.


## 🛠 Setup and Installation
### 🔹 Prerequisites
Ensure you have the following installed:
- Python 3.10+
- GPU

### 🔹 Installation Steps
#### 1. Clone the repository
```bash
git clone https://github.com/Trick5t3r/Finetune-LLM-CASS.git
cd Finetune-LLM-CASS
```
#### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: env\Scripts\activate
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```


## 🚀 Usage
### 1️⃣ Preprocess the dataset
```bash
cd data
unzip cleaned_files_llm.zip
cd ..
```

### 2️⃣ Fine-tune the model
Train the model using the following command:
```bash
python Train/train_sml_last_version.py --model_name t5-base --nb_epoch 4 --summary_type reference_summary --save_path ./outputs/models/finetuned_sml
```

### 3️⃣ Evaluate the model
After training, evaluate performance:
```bash
python evaluate.py --model fine_tuned_model --dataset data/test
```

## Contributor
- Mathias PEREZ
- Théo LE PENDEVEN

## 🔗 References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Google Colab](https://colab.research.google.com/)
- [AWQ Quantization](https://github.com/mit-han-lab/llm-awq)