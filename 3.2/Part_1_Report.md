# Part 1: Environment Setup & Mini Project

## Group Members
- Ahammad, Jony
- Kumara, Muditha
- Podkorytov, Nikolai
- Shrestha Bata, Prabesh

---

## 1. Framework Choice & Installation

### Which Framework Was Selected
**Framework:** [TensorFlow / PyTorch] _(Choose one or both)_

**Reason for Selection:**
[Explain why you chose this framework. Consider factors like ease of use, community support, documentation, project requirements, etc.]

### Installation Method
**Method Used:** [pip / conda / Docker]

**Commands Executed:**
```bash
# Your installation commands here
```

**Installation Status:** ✅ Successful / ⚠️ Issues encountered

---

## 2. Virtual Environment Setup

### Environment Configuration
**Virtual Environment Tool:** venv

**Setup Commands:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

![alt text](image-1.png)

### Environment Details
- **Python Version:** 3.10.12
- **Framework Version:** TensorFlow 2.21.0, PyTorch 2.11.0+cu130

![alt text](image-2.png)

- **Dependencies Installed:** [List key packages]

![alt text](image-3.png)

![alt text](image-4.png)

---

## 3. GPU Verification & Configuration

### GPU Detection Code

```python
"""GPU detection script for TensorFlow and PyTorch."""

def check_tensorflow_gpu():
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        print('TensorFlow GPUs:', gpus)
        print('TensorFlow GPU count:', len(gpus))
    except Exception as e:
        print('TensorFlow error:', e)


def check_pytorch_gpu():
    try:
        import torch

        print('PyTorch CUDA available:', torch.cuda.is_available())
        print('PyTorch CUDA version:', torch.version.cuda)
        print('PyTorch GPU count:', torch.cuda.device_count())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f'GPU {i}:', torch.cuda.get_device_name(i))
    except Exception as e:
        print('PyTorch error:', e)


if __name__ == '__main__':
    print('=' * 50)
    print('GPU DETECTION')
    print('=' * 50)

    check_tensorflow_gpu()
    print()
    check_pytorch_gpu()

```

**Output Screenshot:**
![alt text](image-6.png)

### GPU Status
- **GPU Available:** ✅ Yes 
- **Number of GPUs Detected:** 1
- **GPU Name(s):** NVIDIA GeForce MX350
- **CUDA Version:** 13.0 (PyTorch build)
- **Compatibility Note:** ⚠️ GPU compute capability is sm_61, but the installed PyTorch build supports sm_75 and above.

### Any Issues Encountered & Resolution
- **Issue 1 (PyTorch GPU compatibility):** PyTorch detected the MX350 GPU, but raised warnings that CUDA capability sm_61 is not compatible with the installed build.
    - **Resolution:** Use a PyTorch build/version that supports sm_61, or run PyTorch on CPU in the current setup.

---

## 4. Minimal Test Script

### Test Code Implementation

#### TensorFlow Example
```python
[Paste your minimal TensorFlow test code here]
```

#### PyTorch Example (if applicable)
```python
[Paste your minimal PyTorch test code here]
```

### Test Execution Results

**Console Output Screenshot:**
> **[SCREENSHOT NEEDED]** - Paste screenshot showing:
> - Model summary/architecture
> - Prediction output
> - GPU usage confirmation

**Test Results Summary:**
- ✅ Model creation: Success
- ✅ Predictions generated: Success
- ✅ GPU utilized: Yes / No
- ✅ No errors: Yes / No

---

## 5. Mini Project - Simple Model Implementation

### Project Overview
**Task:** Build and run a minimal deep learning model to confirm the setup is working end-to-end.

### Implementation Details

#### Data Preparation
```python
[Paste code for creating/loading dummy data]
```

#### Model Definition
```python
[Paste code for defining your model]
```

**Model Architecture Screenshot:**
> **[SCREENSHOT NEEDED]** - Paste `model.summary()` output or model visualization

#### Training & Testing
```python
[Paste code for training/testing the model]
```

**Training Output Screenshot:**
> **[SCREENSHOT NEEDED]** - Paste console output showing:
> - Training progress
> - Loss values
> - Accuracy metrics
> - Training time
> - Any warnings or errors

---

## 6. Challenges & Solutions

### Challenge 1
**Problem:** [Describe the issue]

**Solution:** [How it was resolved]

### Challenge 2
**Problem:** [Describe the issue]

**Solution:** [How it was resolved]

---

## 7. Environment Documentation

### requirements.txt
```
[Paste your requirements.txt file content here]
```

### Environment Reproducibility
**How to reproduce this environment:**
```bash
# Step-by-step commands
1. python -m venv myenv
2. source myenv/bin/activate  # On Windows: myenv\Scripts\activate
3. pip install -r requirements.txt
```

---

## 8. Summary & Key Takeaways

### What Was Learned
- [Key learning point 1]
- [Key learning point 2]
- [Key learning point 3]

### Setup Verification Checklist
- ✅ Virtual environment created
- ✅ Framework installed successfully
- ✅ GPU detected (or CPU confirmed as fallback)
- ✅ Minimal test script runs without errors
- ✅ Mini project completed

### Next Steps
[Briefly mention what's planned for Part 2 and Part 3]

---

## 9. Individual Reflections

### Team Member: [Name]
**What went well:**
[Your reflection]

**Challenges faced:**
[Your reflection]

**Questions or concerns:**
[Your reflection]

---

### Team Member: [Name]
**What went well:**
[Your reflection]

**Challenges faced:**
[Your reflection]

**Questions or concerns:**
[Your reflection]

---

### Team Member: [Name]
**What went well:**
[Your reflection]

**Challenges faced:**
[Your reflection]

**Questions or concerns:**
[Your reflection]

---

### Team Member: [Name]
**What went well:**
[Your reflection]

**Challenges faced:**
[Your reflection]

**Questions or concerns:**
[Your reflection]

---

## Appendix: Code Repository
**GitHub/Repository Link:** [Add link if applicable]

**All scripts used:**
- `test_setup.py` - GPU/framework verification
- `mini_project.py` - Minimal model implementation
- `requirements.txt` - Dependency list

