# English ↔ Chuvash — README (concise)

## Files
- english.txt — English sentences (one per line).
- chuvash.txt — Chuvash sentences (one per line, aligned with english.txt).
- dataset_gen.py — data utilities: select / bulk-translate (Yandex) / post-fix (OpenRouter).
- mt.py — train & inference using HF Transformers + bitsandbytes (4‑bit) + PEFT (LoRA).

## Requirements (minimum)
- Python 3.8+, GPU recommended
- pip install torch transformers bitsandbytes peft datasets evaluate sentencepiece sacrebleu scikit-learn pandas requests tqdm numpy

## Environment variables
- HF_TOKEN — (optional) Hugging Face access token for model download.
- YANDEX_API_KEY, YANDEX_FOLDER_ID — for dataset_gen.py translate (Yandex).
- OPENROUTER_API_KEY — for dataset_gen.py fix (OpenRouter).

## mt.py — quick summary
- Model: google/t5gemma-2-4b-4b
- Prefix: "Translate English to Chuvash: "
- Uses BitsAndBytes 4‑bit quantization, prepare_model_for_kbit_training and LoRA (r=16, alpha=32, dropout=0.05).
- Builds HF Dataset from parallel files, tokenizes (max 256), trains with Seq2SeqTrainer and computes sacreBLEU.
- Default OUTPUT_DIR: t5gemma2-en-chv-qlora

## Common commands
- Train:
  python mt.py train english.txt fixed_chuvash.txt
- Translate (inference with adapter):
  python mt.py translate --in to_translate.txt --out translations.txt --adapter t5gemma2-en-chv-qlora/final --batch 8

## Key training defaults
- batch_size per device = 8, grad_accum = 16, lr = 2e-4, epochs = 3, max length = 256

## dataset_gen.py — quick summary
- select: pick candidates similar to a test set (TF-IDF + nearest neighbors).
  Example:
  python dataset_gen.py select --test-csv test.csv --out selected.txt --n-neighbors 20 --total-k 8000
- translate: bulk-translate with Yandex (batches & retries).
  Example:
  export YANDEX_API_KEY=...; export YANDEX_FOLDER_ID=...
  python dataset_gen.py translate --in english.txt --out chuvash.txt
- fix: post-edit translations via OpenRouter (forces JSON reply with fixed_translation + issues).
  Example:
  export OPENROUTER_API_KEY=...
  python dataset_gen.py fix --en english.txt --cv chuvash.txt --out fixed_chuvash.txt --report fix_report.csv

## Quick workflow
1. Produce raw translations (optional): dataset_gen.py translate -> chuvash.txt  
2. Post-edit (optional): dataset_gen.py fix -> fixed_chuvash.txt + fix_report.csv  
3. Train: python mt.py train english.txt fixed_chuvash.txt  
4. Inference: python mt.py translate --in new_en.txt --out new_cv.txt --adapter <adapter_path>

## Notes & caveats
- english.txt and chuvash.txt must be aligned and UTF‑8 encoded.
- Yandex/OpenRouter calls consume quota/cost—use batching and --max-rows to limit usage.
- bitsandbytes 4‑bit + PEFT require compatible CUDA / driver / bitsandbytes build.
- Check CLI flags with -h for exact names (some script copies may have minor formatting issues).
- If OOM: reduce per-device batch size or increase gradient accumulation.
