# Multi-lingual Article Translation: Project Document

Oct 12, 2026 · @Godfred Opintan

## Overview

This project fine-tunes NLLB-200-distilled-600M with LoRA on a free Colab T4 to translate articles between English and French, Arabic, and Swahili, then serves it in a Gradio app.

**Central question:** does fine-tuning on article-style data beat a strong zero-shot baseline, and for which languages?

## Objectives

- Fine-tune a multilingual translation model, with hyperparameter optimization (HPO).
- Evaluate it with automatic metrics and an LLM-as-judge rating.
- Build an app that translates whole articles, not just sentences.
- Package the code so later projects can reuse it.

## Success criteria

- A results table of zero-shot vs. fine-tuned scores for all six directions, with an honest account of where tuning helped and where it did not.
- A public app on Hugging Face Spaces and a model card on the Hub.
- A repo where a new project needs a new config file, not a new notebook.

## Scope

The core scope is six English-centric directions: EN↔FR, EN↔AR, and EN↔SW. The full set of 12 directions across four languages is too much for a free Colab.

| Item | In scope | Out of scope (or bonus) |
| --- | --- | --- |
| Languages | English, French, Arabic, Swahili | Any other language |
| Directions | EN↔FR, EN↔AR, EN↔SW | Direct FR/AR/SW pairs (bonus: pivot through English) |
| Text unit | Sentences, with article-level splitting and reassembly | Document-level context modelling |
| Input | Pasted text and `.txt` upload | PDFs, web pages, images |

Models translate sentences, not articles, so the pipeline splits an article into sentences or paragraphs, translates each, and reassembles them. Building and testing that step is part of the project.

## Compute constraints

The only compute is a free Google Colab, so every choice below is sized for a T4 GPU (16 GB VRAM) and roughly 12 GB of system RAM. Sessions disconnect after a few hours or on inactivity, and GPU availability is not guaranteed.

| Constraint | Consequence |
| --- | --- |
| 16 GB VRAM | Fine-tune the 600M model with LoRA; run the 1.3B model zero-shot only |
| No bf16 on T4 | Use fp16 with care; fall back to fp32 base weights if losses turn NaN |
| Session disconnects | Save checkpoints to Google Drive or the Hub every few hundred steps; keep every notebook resumable |
| Optuna state | Store the HPO study in SQLite on Drive so trials survive a disconnect |
| CPU-only free Spaces | Merge the adapter, keep inputs short, and optionally quantize (CTranslate2 int8) |

## Model

The base model is NLLB-200-distilled-600M, because one multilingual model covers all four languages, including Swahili, and fits a free T4 with LoRA.

| Option | Verdict |
| --- | --- |
| NLLB-200-distilled-600M | **Chosen.** Covers all four languages; trainable with LoRA on a T4 |
| NLLB-200 1.3B | Zero-shot baseline only; too heavy to fine-tune here |
| M2M-100 (418M) | Older, weaker on Swahili |
| mBART-50 | Decent, but thinner Swahili support |
| Opus-MT (MarianMT) | Small and fast, but one model per direction, which suits a single multilingual app poorly |
| Small multilingual LLM (Qwen, Llama, Gemma) | More flexible, but heavier to fine-tune and serve |

NLLB is already strong, so the zero-shot score is the baseline to beat. Expect the largest gains where data is scarce or domain-specific, most likely Swahili news text.

## Data

Training uses 10–20k cleaned sentence pairs per direction, mixed into one set, and FLORES-200 is held out strictly for evaluation. Dataset names below come from planning, so confirm availability and licenses on the Hub before use.

| Purpose | Dataset |
| --- | --- |
| General training | OPUS-100, Tatoeba, News Commentary |
| News-domain training (EN/FR↔SW) | MAFAND-MT (Masakhane) |
| Held-out evaluation | FLORES-200 devtest |

### Cleaning steps

1. Deduplicate pairs.
2. Filter by source/target length ratio.
3. Check language ID on both sides.
4. Cap sequences at 128 tokens.

### Rules

- Never let FLORES text enter training, so evaluation stays honest.
- Record every dataset's license in the README.
- Mix all directions in training to limit catastrophic forgetting.

## Fine-tuning and HPO

Training uses LoRA through PEFT with `Seq2SeqTrainer`, and HPO runs 8–12 short Optuna trials on a small subset before one final run on the full set.

| Setting | Plan |
| --- | --- |
| Method | LoRA adapters on NLLB-600M |
| Framework | Hugging Face `Seq2SeqTrainer`, PEFT, Optuna via `Trainer.hyperparameter_search` |
| HPO trials | 8–12, each about 300–500 steps on a \~5k-pair subset |
| Search space | Learning rate (1e-5 to 1e-4, log scale), LoRA rank, LoRA alpha, warmup ratio |
| Final run | Best config on the full training set, 1–3 epochs |
| Tracking | Weights & Biases or TensorBoard |
| Storage | Optuna SQLite study and checkpoints on Google Drive |

Batch size and weight decay are left out of the search because they matter less at this scale. The study can run across two or three Colab sessions.

## Evaluation

Evaluation has three layers, so no single score decides the outcome, and every layer compares the zero-shot baseline with the fine-tuned model in all six directions.

| Layer | Method | Size |
| --- | --- | --- |
| 1. Automatic metrics | chrF++ (better than BLEU for Arabic and Swahili); COMET if Colab allows | Full FLORES-200 devtest |
| 2. LLM-as-judge | 1–5 scores for adequacy, fluency, and terminology/named entities; fixed rubric, JSON output | About 100 samples per direction (600 total) |
| 3. Human sanity check | Native-speaker review; check that judge scores correlate with chrF++/COMET | About 30 samples per language, if a reviewer is available |

### Judge protocol

- Give the judge both source and candidate, with a fixed rubric.
- When comparing baseline and fine-tuned output, randomize their order.
- Cache every judge response to disk to avoid paying or waiting twice.
- Use a free-tier or low-cost API, and check current limits before relying on them.

LLM judges can be less reliable on low-resource languages such as Swahili, so the human check and the correlation test matter most there.

## Application

The app is a Gradio interface on Hugging Face Spaces that loads the merged model from the Hub and translates whole articles while keeping paragraph structure.

### Core features

- Paste text or upload a `.txt` file.
- Auto-detect the source language (fastText LID or `langdetect`), with a manual override.
- Choose a target language from the four supported.
- Split into paragraphs and sentences, translate in batches, and reassemble.
- Show Arabic output right-to-left.
- Enforce a length limit to keep CPU latency acceptable.
- Offer a "pivot via English" option for non-English pairs.

### Serving notes

- A free Space runs on CPU, so merge the LoRA adapter into the base weights before deploying.
- If latency is a problem, quantize with CTranslate2 int8.
- Stretch: side-by-side baseline vs. fine-tuned output.

## Reusable scaffolding

Logic lives in `src/` on GitHub and notebooks stay thin, so Colab clones the repo each session and a new project needs only a new config.

```text
project/
├── configs/      # YAML: model, data, LoRA, HPO space
├── src/
│   ├── data/     # dedupe, length-ratio filter, language ID
│   ├── train.py  # config-driven LoRA training + Optuna HPO
│   └── eval/     # metrics + LLM judge with disk caching
├── notebooks/    # thin runners that call src/
├── app/          # Gradio app, model ID read from config
└── README.md     # results table, model card, dataset licenses
```

- **Config-driven training:** one YAML file drives a single `train.py`.
- **Evaluation harness:** takes a model and a test set, returns metrics and judge scores.
- **Data module:** importable cleaning functions.
- **Deployment template:** swapping the model is a one-line config change.

## Dependencies

The project needs a small Python stack and five external services. Pin exact versions in `requirements.txt` after the first successful baseline run, because Colab's preinstalled packages change over time.

### Python packages

| Package | Used for | Phase |
| --- | --- | --- |
| `torch` | Model runtime | All |
| `transformers` | NLLB model, tokenizer, `Seq2SeqTrainer` | All |
| `sentencepiece` | NLLB tokenization | All |
| `datasets` | Loading OPUS-100, MAFAND-MT, FLORES-200 | 1 |
| `fasttext` or `langdetect` | Language ID for cleaning and the app | 1, 4 |
| `peft` | LoRA adapters | 2–3 |
| `accelerate` | GPU training and mixed precision | 2–3 |
| `optuna` | HPO study with SQLite storage | 2 |
| `wandb` or `tensorboard` | Experiment tracking | 2–3 |
| `sacrebleu` | chrF++ scoring | 1, 3 |
| `unbabel-comet` | COMET scoring (optional) | 1, 3 |
| `pyyaml` | Config files | All |
| `gradio` | Web app | 4 |
| `ctranslate2` | int8 quantization (optional stretch) | 4 |

### External services

| Service | Used for |
| --- | --- |
| Google Colab (free T4) | All training and evaluation |
| Google Drive | Checkpoints and the Optuna SQLite study |
| GitHub | Code repo, cloned at the start of each session |
| Hugging Face Hub and Spaces | Datasets, model and adapter hosting, the app |
| Judge LLM API (free-tier or low-cost) | LLM-as-judge scoring |

## Work plan

Work runs in four phases, about four to six Colab sessions of GPU time in total, and each ends in a gatec.

[embedded content: work plan · 4 phases, 4 gates\]

No deadline is set, so phases are ordered by dependency rather than dated; the app and write-up take the remaining time.

## Risks and mitigations

The main risks are weak Swahili quality, limited compute, and forgetting across directions.

| Risk | Mitigation |
| --- | --- |
| Low-resource quality in Swahili | Lean on MAFAND-MT news data; report results honestly if gains are small |
| Compute limits and disconnects | LoRA, small data subsets, checkpoints on Drive, resumable notebooks |
| Catastrophic forgetting | Mix all directions in one training set; compare against the zero-shot baseline |
| Sentence-splitting errors hurt long articles | Test the splitter on real articles early |
| LLM judge unreliable on Swahili | Native-speaker spot check and correlation with chrF++/COMET |
| Dataset license problems | Check and record licenses before training |
| FLORES leakage into training | Exclude FLORES from every training source and verify before evaluating |

## Deliverables and stretch goals

### Deliverables

- [ ] Public GitHub repo with configs, `src/`, thin notebooks, and a README
- [ ] LoRA adapter and merged model on the Hugging Face Hub, with a model card
- [ ] Results table: zero-shot vs. fine-tuned, chrF++ and LLM-judge scores, six directions
- [ ] Gradio app live on Hugging Face Spaces
- [ ] Short write-up of findings, including where fine-tuning did not help

### Stretch goals

- [ ] Run NLLB 1.3B zero-shot as an extra baseline
- [ ] Quantize with CTranslate2 int8 for faster CPU serving
- [ ] Side-by-side baseline vs. fine-tuned output in the app
- [ ] Report bonus pivot-through-English results for FR/AR/SW pairs
