# 🧴 DermBench

![DermBench overview](assets/introduction.png)

DermBench is the public benchmark and evaluator repository for **Towards Trustworthy Dermatology MLLMs: A Benchmark and Multimodal Evaluator for Diagnostic Narratives**. It supports dermatology image-to-diagnostic-narrative evaluation with fixed prompts, clinician-certified reference CoT texts, DermEval reference-free evaluation utilities, and six clinical scoring dimensions.

This repository contains the DermBench text release and minimal DermEval code needed to reproduce the benchmark interfaces. It does **not** redistribute DermNet images.

## 📦 Contents

- `data/`: 4000 CoT/reference texts, manifest, category counts, and source audit.
- `dermbench/`: lightweight Python helpers, benchmark prompts/config, and DermEval utilities.
- `scripts/`: validation and optional local evaluation utilities.
- `docs/`: data mapping, scoring criteria, DermNet notice, and DermEval guide.
- `assets/`: README figure.

The CoT files were prepared from the source text tree `Dermnet2/step4_txt/test`. Candidate model outputs, result files, and legacy runner scripts are intentionally excluded.

## 🧭 Data Mapping

Each JSONL row has this shape:

```json
{
  "id": "dermbench_0001",
  "source": "DermNet test set",
  "split": "test",
  "category": "Acne and Rosacea Photos",
  "diagnosis": "PerioralDermEye",
  "dermnet_image_path": "Acne and Rosacea Photos/07PerioralDermEye.jpg",
  "cot": "data/cot/Acne and Rosacea Photos/07PerioralDermEye.jpg.txt",
  "cot_sha256": "..."
}
```

The corresponding image is expected at:

```text
$DERMNET_TEST_ROOT/<dermnet_image_path>
```

For example:

```text
$DERMNET_TEST_ROOT/Acne and Rosacea Photos/07PerioralDermEye.jpg
```

DermNet images are external to this repository. Obtain and use them only under the applicable DermNet terms and any separate research agreement that applies to your project.

Official DermNet links:

- DermNet home: https://dermnetnz.org/
- DermNet image library: https://dermnetnz.org/images
- DermNet image licence: https://dermnetnz.org/image-licence
- DermNet website terms: https://dermnetnz.org/terms

## 📏 Scoring Metrics

DermBench scores each candidate narrative on six 0-5 dimensions:

- Accuracy
- Safety
- Medical Groundedness
- Clinical Coverage
- Reasoning Coherence
- Description Precision

The metric order is fixed in `dermbench/configs/models.json` and `dermbench/scoring.py`.

## ⚙️ Setup

```bash
conda env create -f environment.yml
conda activate dermbench
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Validate the text-only release:

```bash
python scripts/validate_release.py --check-sha
```

Validate mapping against a local DermNet test image root:

```bash
export DERMNET_TEST_ROOT=/your/dermnet/test/root
python scripts/validate_release.py --image-root "$DERMNET_TEST_ROOT"
```

## 🧪 DermEval

DermEval is the reference-free evaluator component described in the paper. The repository includes its six-dimension rubric, prompt builder, deterministic score parser, raw-record conversion helper, and an optional local inference entrypoint for compatible LLaVA-style checkpoints.

Core utilities can be used without any API credentials:

```bash
python -m dermbench.dermeval.score_parser path/to/dermeval_output.txt
python -m dermbench.dermeval.convert_to_llava_sft --input data/dermeval_records.jsonl --output outputs/dermeval_sft.json
```

See `docs/dermeval.md` for the record schema, output format, and optional local inference notes.

## 📄 License

The DermBench text release and repository code are distributed under the Creative Commons Attribution-NonCommercial 4.0 International license. Commercial use requires separate permission.

This license does not cover DermNet images. See `docs/data_notice.md` for DermNet image links and usage notes.

## 📚 Citation

```bibtex
@article{shen2025towards,
  title={Towards Trustworthy Dermatology MLLMs: A Benchmark and Multimodal Evaluator for Diagnostic Narratives},
  author={Shen, Yuhao and Qian, Jiahe and Zhang, Shuping and Chen, Zhangtianyi and Lu, Tao and Zhou, Juexiao},
  journal={arXiv preprint arXiv:2511.09195},
  year={2025}
}
```
