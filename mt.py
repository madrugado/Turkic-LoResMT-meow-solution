import os
import argparse
import numpy as np
import torch
import bitsandbytes as bnb

from datasets import Dataset
import evaluate

from transformers import (
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

MODEL_ID = "google/t5gemma-2-4b-4b"
PREFIX = "Translate English to Chuvash: "
OUTPUT_DIR = "t5gemma2-en-chv-qlora"


def read_parallel(src_path, tgt_path):
    src = open(src_path, encoding="utf-8").read().splitlines()
    tgt = open(tgt_path, encoding="utf-8").read().splitlines()
    if len(src) != len(tgt):
        raise ValueError(f"line mismatch: {len(src)} vs {len(tgt)}")
    return [(s.strip(), t.strip()) for s, t in zip(src, tgt) if s.strip() and t.strip()]


def find_lora_targets(model):
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                targets.add(leaf)
    return sorted(targets)


def train(args):
    hf_token = os.environ.get("HF_TOKEN")
    pairs = read_parallel(args.src, args.tgt)
    print(f"pairs: {len(pairs)}")

    ds = Dataset.from_dict({
        "text": [PREFIX + s for s, _ in pairs],
        "target": [t for _, t in pairs],
    }).train_test_split(test_size=0.01, seed=42)

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer = processor.tokenizer

    def preprocess(batch):
        return tokenizer(batch["text"], text_target=batch["target"], max_length=256, max_target_length=256,
                         truncation=True)

    train_tok = ds["train"].map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    eval_tok = ds["test"].map(preprocess, batched=True, remove_columns=ds["test"].column_names)

    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=hf_token, device_map="auto",
                                                  quantization_config=qconfig, torch_dtype=torch.float16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    target_modules = find_lora_targets(model)
    print(f"lora targets: {target_modules}")

    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=32,
                             lora_dropout=0.05, target_modules=target_modules, bias="none")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, padding=True, label_pad_token_id=-100,
                                      pad_to_multiple_of=8)
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = sacrebleu.compute(predictions=[p.strip() for p in decoded_preds],
                                   references=[[l.strip() for l in decoded_labels]])
        return {"sacrebleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=False,
        generation_max_length=256,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_tok, eval_dataset=eval_tok,
                             data_collator=collator, processing_class=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")


def translate(args):
    hf_token = os.environ.get("HF_TOKEN")
    adapter_path = args.adapter or f"{OUTPUT_DIR}/final"

    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=hf_token, device_map="auto",
                                                  quantization_config=qconfig, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer = processor.tokenizer

    lines = open(args.inp, encoding="utf-8").read().splitlines()
    results = []

    for i in range(0, len(lines), args.batch):
        batch = [PREFIX + t for t in lines[i:i + args.batch]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    with open(args.out, "w", encoding="utf-8") as f:
        for t in results:
            f.write(t + "\n")
    print(f"wrote {args.out} ({len(results)} lines)")


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)

p1 = sp.add_parser("train")
p1.add_argument("--src", default="english.txt")
p1.add_argument("--tgt", default="fixed_chuvash.txt")

p2 = sp.add_parser("translate")
p2.add_argument("--in", dest="inp", required=True)
p2.add_argument("--out", required=True)
p2.add_argument("--adapter", default=None)
p2.add_argument("--batch", type=int, default=8)

args = ap.parse_args()
{"train": train, "translate": translate}[args.cmd](args)
