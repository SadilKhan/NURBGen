<div align="center">

# NURBGen
### High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue?style=flat-square)](https://ojs.aaai.org/index.php/AAAI/article/view/37922)
[![arXiv](https://img.shields.io/badge/arXiv-2511.06194-b31b1b?style=flat-square)](https://arxiv.org/abs/2511.06194)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=flat-square)](https://muhammadusama100.github.io/NURBGen/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PartABC-orange?style=flat-square)](https://huggingface.co/datasets/SadilKhan/PartABC)
[![Model](https://img.shields.io/badge/🤗%20Model-NURBGen-yellow?style=flat-square)](https://huggingface.co/SadilKhan/NURBGen)
[![Poster](https://img.shields.io/badge/Poster-PDF-purple?style=flat-square)](https://mdsadilkhan.onrender.com/publications/data/nurbgen_poster.png)

**The first framework to generate industry-standard NURBS surfaces directly from text prompts,
producing editable, parametric CAD models convertible to STEP format.**

[Muhammad Usama](https://muhammadusama100.github.io/MUsama/)<sup>1,2,3 \*</sup> ·
[Mohammad Sadil Khan](https://mdsadilkhan.onrender.com/)<sup>1,2,3 \* †</sup> ·
[Didier Stricker](https://av.dfki.de/members/stricker/)<sup>1,2</sup> ·
[Muhammad Zeshan Afzal](https://scholar.google.com/citations?user=kHMVj6oAAAAJ&hl=en)<sup>1,3</sup>

<sup>1</sup>DFKI Kaiserslautern &nbsp;·&nbsp; <sup>2</sup>RPTU Kaiserslautern-Landau &nbsp;·&nbsp; <sup>3</sup>MindGauge

<sup>\* Equally contributing first authors &nbsp;·&nbsp; † Corresponding author</sup>

</div>

---

## Overview

NURBGen is the **first LLM-based framework** for generating industry-standard NURBS (Non-Uniform Rational B-Spline) representations directly from natural language. Unlike mesh- or voxel-based approaches, NURBGen produces fully parametric, editable CAD models exportable to STEP format, the standard for engineering workflows.

- 🏆 **64.1% human preference** over all baselines (5 CAD designers, 1k samples)
- 📐 **Lowest invalidity ratio** of 0.018 — valid BRep structures
- 🗃️ **partABC dataset** — 300k part-level CAD models with NURBS annotations and auto-generated captions (~85% accuracy)
- 🔀 **Hybrid representation** combining untrimmed NURBS with analytic primitives for geometric robustness

---

## The partABC Dataset

[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-PartABC-orange?style=flat-square)](https://huggingface.co/datasets/SadilKhan/PartABC)

Derived from the ABC dataset (1M CAD models), partABC decomposes assembly-level designs into individual part-level components using PythonOCC, generating 3M instances. A complexity-aware filtering strategy retains **300k geometrically diverse parts**.

| Property | Value |
|---|---|
| Total parts | 300,000 |
| Caption accuracy | ~85% |
| Complexity tiers | Simple (10%) / Moderate (50%) / Complex (40%) |
| Captioning model | InternVL3-13B |
| Render setup | 4 orthographic views at 512×512 via Blender + Freestyle edges |


---

## How to Use

### Installation

```bash
pip install ms-swift transformers peft torch
```

### Single Prompt (CLI)

```bash

# For single prompt, use the --prompt flag:
python -n src.infer_nurbgen --prompt "Socket head cap screw with a large countersunk washer. Features a hexagonal socket drive and a cylindrical threaded shank. Dimensions: length 92.96 mm, width 79.38 mm, height 43.66 mm. Ensure smooth curvature at transitions." --output_dir ./results

# For batch processing with file outputs, create a text file (e.g., prompts.txt) with one prompt per line:
python -n src.infer_nurbgen --input prompts.txt --output_dir ./results

# For json input [{"uid", "caption"},{"uid", "caption"}], create a jsonl file (e.g., prompts.jsonl):
python -n src.infer_nurbgen --input prompts.jsonl --output_dir ./results
```

### Python API — ms-swift

```python
from swift.llm import PtEngine, RequestConfig, InferRequest

engine = PtEngine(
    "Qwen/Qwen3-4B",
    adapters=["SadilKhan/NURBGen"],
    use_hf=True,
)

response = engine.infer(
    [InferRequest(messages=[{"role": "user", "content": "Generate NURBS for the following: Design a rectangular plate with dimensions 330.20 mm x 233.40 mm x 6.00 mm. Include two square through-holes near each end."}])],
    request_config=RequestConfig(max_tokens=8192, temperature=0.3),
)
print(response[0].choices[0].message.content)
```

### Python API — HuggingFace / PEFT

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "SadilKhan/NURBGen")
model.eval()

messages = [{"role": "user", "content": "Generate NURBS for the following: Design a rectangular plate with dimensions 330.20 mm x 233.40 mm x 6.00 mm. Include two square through-holes near each end. "}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=8192, do_sample=False)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--prompt` / `-p` | — | Inline single prompt string |
| `--input` / `-i` | — | Path to `.txt`, `.json`, or `.jsonl` |
| `--output_dir` / `-o` | `./nurbgen_outputs` | Directory for output `.txt` files |
| `--batch_size` | `4` | Prompts per inference batch |
| `--max_new_tokens` | `8192` | Maximum tokens to generate |
| `--temperature` | `0.3` | `0` = greedy decoding |
| `--save_summary` | off | Also write `results_summary.json` |

---

## Citation

If you find NURBGen useful in your research, please cite:

```bibtex
@inproceedings{usama2025nurbgen,
  title     = {NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling},
  author    = {Usama, Muhammad and Khan, Mohammad Sadil and Stricker, Didier and Afzal, Muhammad Zeshan},
  booktitle = {AAAI},
  year      = {2026}
}
```

---

<div align="center">
  <sub>Developed at DFKI Kaiserslautern · MindGarage · RPTU Kaiserslautern-Landau</sub><br>
  <sub>Contact: <a href="mailto:mdsadilkhan99@gmail.com">mdsadilkhan99@gmail.com</a></sub>
</div>