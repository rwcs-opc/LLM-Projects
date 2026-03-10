# 🧠 LLM Fine-Tuning with QLoRA

This project demonstrates a **clean end-to-end workflow for fine-tuning a Large Language Model (LLM)** using **QLoRA (Quantized Low-Rank Adaptation)**.

The goal is to train an instruction-following model efficiently using **low GPU memory** environments such as **A100 MIG or consumer GPUs**.

---

# 📂 Project Structure

```
LLM-Fine-Tunning-aikosh-test1
│
├── LLM_Fine_Tuning_QLoRA_Clean.ipynb
├── README.md
└── lora-adapter/   (generated after training)
```

---

# 🚀 Features

* Load LLM in **4-bit quantization**
* Fine-tune with **LoRA / QLoRA**
* Train using **HuggingFace Trainer**
* Save **LoRA adapters only**
* Run inference with **controlled stop tokens**

---

# 🧰 Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* PEFT (LoRA)
* BitsAndBytes (4-bit quantization)
* HuggingFace Datasets

---

# ⚙️ Installation

Install required dependencies:

```bash
pip install \
torch==2.1.2 \
transformers==4.36.2 \
accelerate==0.25.0 \
peft==0.7.1 \
bitsandbytes==0.41.3 \
datasets
```

---

# 📥 Base Model

This project uses the following base model:

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The model is loaded in **4-bit NF4 quantization** to reduce GPU memory usage.

---

# 🧠 QLoRA Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

---

# 🔧 LoRA Adapter Setup

LoRA adapters are attached to attention layers.

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

Only the **LoRA parameters are trained**, while the base model remains frozen.

---

# 📊 Dataset Format

Training data is structured as **Instruction → Response** format.

Example:

```
### Instruction:
Explain JWT

### Response:
JWT is a compact, URL-safe token used for authentication and authorization.
```

Dataset is loaded using **HuggingFace Datasets**.

---

# 🔤 Tokenization

The dataset is tokenized and truncated to **512 tokens**.

```python
tokenizer(
    example["text"],
    truncation=True,
    padding="max_length",
    max_length=512
)
```

---

# 🏋️ Training

Training is performed using **HuggingFace Trainer**.

Key parameters:

```
Batch size: 1
Gradient accumulation: 8
Epochs: 3
Learning rate: 2e-4
Precision: BF16
```

Output directory:

```
./llm-output
```

---

# 💾 Saving Model

Only the **LoRA adapters** are saved.

```python
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")
```

This keeps the model lightweight and reusable.

---

# 🔎 Inference

After training, the model can generate responses using prompt instructions.

Stop tokens can be used to control generation.

Example stop token:

```
\end
```

---

# ⚠️ Important Notes

* Do **not merge LoRA adapters** with the base model when using 4-bit quantization.
* LoRA adapters provide **modular and efficient fine-tuning**.
* For merging, reload the base model in **FP16** first.

---

# 🎯 Use Cases

* Domain-specific chatbots
* Instruction-tuned assistants
* Knowledge-based AI tools
* Custom LLM experimentation

---

# 📚 References

* HuggingFace Transformers
* PEFT (LoRA)
* BitsAndBytes
* QLoRA Research

---

# 👨‍💻 Author

Rustam Ahmed
Director – RWorld Computer Solutions Pvt Ltd

---
