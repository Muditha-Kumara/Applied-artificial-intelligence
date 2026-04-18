# LLM Configurations, Tooling & Chatbot APIs

## 1. Overview

In this part of the project, our group explored how large language models function as configurable inference systems rather than simple text‑completion tools. The notebook introduced an engineering‑oriented perspective on LLM behavior, focusing on how output quality depends on prompt content, model weights, decoding strategies, and application‑level logic.

We worked with open‑source chat models from Hugging Face, experimented with decoding parameters, inspected chat templates, and used an interactive playground to observe how configuration changes influence model behavior.

---

## 2. Key Concepts Explored

### 2.1 LLM Behavior Pipeline
We learned that LLM output is shaped by four interacting layers:

1. **Prompt content** – instructions, examples, formatting  
2. **Model weights** – capabilities learned during training  
3. **Decoding strategy** – temperature, top‑p, top‑k, repetition penalty  
4. **Application logic** – memory, tools, retrieval, stop rules, UI  

This helped us understand that unexpected behavior often comes from configuration issues rather than the model itself.

### 2.2 Chat Templates
We inspected how the tokenizer formats messages before tokenization using:

tokenizer.apply_chat_template(...)

Different models use different wrappers, so the same messages may be interpreted differently depending on the template.

### 2.3 System Instructions
We experimented with three ways of providing system instructions:

- manual text  
- local file path  
- uploaded `.txt` or `.md` file  

The notebook resolves them in this priority:

**uploaded file → local file → manual text**

### 2.4 Model Catalog and Loading
We explored a curated list of models, including:

- Qwen2.5‑3B‑Instruct  
- Mistral‑7B‑Instruct  
- Phi‑3.5‑mini  
- Llama‑3.2‑3B‑Instruct  
- Gemma‑2‑2B‑it  
- DeepSeek‑R1‑Distill‑Qwen‑7B  

We used widgets to select a model, enable local‑files‑only mode, try 4‑bit loading, and load the model interactively.

### 2.5 Decoding Parameters
We experimented with:

- temperature  
- top‑p  
- top‑k  
- repetition penalty  
- max new tokens  
- stop sequences  
- random seed  

We observed how each parameter changes creativity, determinism, and stability.

### 2.6 Interactive Playground
The playground allowed us to:

- set system instructions  
- write prompts  
- adjust decoding parameters  
- generate responses  
- inspect token counts  
- preview chat templates  

This helped us understand how small configuration changes affect output quality.

---

## 3. Group Reflection

Working through this notebook gave our group a deeper understanding of how LLMs operate internally. Instead of treating the model as a black box, we learned to think in terms of inference pipelines, configuration layers, reproducibility, and engineering trade‑offs.

**Key insights:**

- Chat templates strongly influence model behavior  
- System instructions should be externalized for consistency  
- Temperature and top‑p have the strongest effect on creativity  
- Repetition penalty reduces looping and redundancy  
- Local inference requires careful hardware and quantization choices  
- The same prompt can produce very different outputs depending on decoding settings  

---

## 4. Challenges Encountered

We encountered several practical issues:

- Some models required enabling `trust_remote_code`  
- 4‑bit loading was not available for all models  
- Certain models required license acceptance on Hugging Face  
- Chat templates differed across models  
- Large models increased load time and memory usage  
- Stop sequences sometimes cut off responses too early  

These challenges helped us understand real‑world constraints of LLM deployment.

---

## 5. Required Screenshots

Below is the list of screenshots we included in the report:

1. **Model Catalog Output**  

2. **Model Loading Widget (before loading)** 

3. **Successful Model Load Output**  

4. **Chat Template Preview Output**  

5. **System Instruction Resolution (manual/file/upload)**  

6. **Playground UI with decoding parameters**  

7. **Run Summary (input/output tokens + generation kwargs)**  

8. **Assistant Response Box**  

9. *(Optional)* Comparison run with different decoding settings  

---
