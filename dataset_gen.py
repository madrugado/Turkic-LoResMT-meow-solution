import argparse
import json
import os
import re
import time
import unicodedata
from time import perf_counter

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MIXED_MAP = str.maketrans({
    "ă": "ӑ", "Ă": "Ӑ", "ĕ": "ӗ", "Ĕ": "Ӗ",
    "ç": "ҫ", "Ç": "Ҫ", "ÿ": "ӳ", "Ÿ": "Ӳ",
})

CHUV_LETTERS = set("ӑӗҫӳӐӖҪӲ")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s.strip())


def fix_script(s):
    return norm((s or "").translate(MIXED_MAP))


def has_chuv(s):
    return any(c in CHUV_LETTERS for c in (s or ""))


def dedup_key(s):
    s = norm(s).lower()
    s = re.sub(r"['']", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def good_sent(s, min_ch, max_ch, min_w):
    s = norm(s)
    if len(s) < min_ch or len(s) > max_ch:
        return False
    if len(s.split()) < min_w:
        return False
    if "http://" in s or "https://" in s:
        return False
    return bool(re.search(r"[A-Za-z]", s))


def run_select(args):
    sources = [
        ("openwebtext", "cands_openwebtext_1000000.txt"),
        ("wiki", "cands_wiki_1000000.txt"),
        ("cc_news", "cands_cc_news_1000000.txt"),
    ]

    df = pd.read_csv(args.test_csv)
    test = [norm(s) for s in df["source_en"].astype(str).tolist()]
    n_test = len(test)

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.95,
        sublinear_tf=True, max_features=250000, dtype=np.float32
    )
    X_t = vec.fit_transform(test)

    best_sim = {}
    canon = {}
    per_test = [[] for _ in range(n_test)]

    for tag, path in sources:
        print(f"\n[{tag}] loading {path}")
        raw = open(path, encoding="utf-8").read().splitlines()

        seen = set()
        cands, keys = [], []
        for line in raw:
            if not good_sent(line, args.min_chars, args.max_chars, args.min_words):
                continue
            k = dedup_key(line)
            if not k or k in seen:
                continue
            seen.add(k)
            cands.append(norm(line))
            keys.append(k)
            if k not in canon:
                canon[k] = norm(line)

        print(f"[{tag}] raw={len(raw)} kept={len(cands)}")

        X_c = vec.transform(cands)
        nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric="cosine", algorithm="brute")
        nn.fit(X_c)

        dist, idx = nn.kneighbors(X_t)
        sim = 1.0 - dist

        for i in range(n_test):
            for j in range(args.n_neighbors):
                cand_idx = int(idx[i, j])
                score = float(sim[i, j])
                key = keys[cand_idx]
                if key not in best_sim or score > best_sim[key]:
                    best_sim[key] = score
                per_test[i].append((score, key))

    print(f"\nUnique: {len(best_sim)}")

    must_keep = set()
    for i in range(n_test):
        hits = sorted(per_test[i], key=lambda x: -x[0])
        kept = 0
        for score, key in hits:
            if args.sim_floor is None or score >= args.sim_floor:
                must_keep.add(key)
                kept += 1
                if kept >= args.cover_per_test:
                    break

    print(f"Must-keep: {len(must_keep)}")

    items = [(k, s) for k, s in best_sim.items() if args.sim_floor is None or s >= args.sim_floor]
    items.sort(key=lambda x: -x[1])

    selected = []
    seen_sel = set()

    for k, s in sorted([(k, best_sim[k]) for k in must_keep], key=lambda x: -x[1]):
        if k not in seen_sel:
            seen_sel.add(k)
            selected.append((k, s))

    for k, s in items:
        if len(selected) >= args.total_k:
            break
        if k not in seen_sel:
            seen_sel.add(k)
            selected.append((k, s))

    selected.sort(key=lambda x: -x[1])

    out_path = args.out or f"selected_top{len(selected)}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for k, _ in selected:
            f.write(canon[k] + "\n")

    sims = np.array([s for _, s in selected])
    print(f"\nFinal: {len(selected)}")
    print(f"sim min/med/max: {sims.min():.3f} / {np.median(sims):.3f} / {sims.max():.3f}")
    print(f"wrote: {out_path}")


def chunk_lines(items, max_texts, max_chars):
    batches = []
    cur, cur_len = [], 0
    for idx, txt in items:
        if cur and (len(cur) >= max_texts or cur_len + len(txt) > max_chars):
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append((idx, txt))
        cur_len += len(txt)
    if cur:
        batches.append(cur)
    return batches


def yandex_batch(texts, api_key, folder_id, src_lang, tgt_lang):
    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}"}
    payload = {"folderId": folder_id, "sourceLanguageCode": src_lang, "targetLanguageCode": tgt_lang, "texts": texts}

    for attempt in range(1, 7):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            out = [t.get("text", "") for t in r.json().get("translations", [])]
            if len(out) != len(texts):
                raise RuntimeError("length mismatch")
            return out
        except Exception:
            if attempt == 6:
                raise
            time.sleep(1.5 * attempt)


def run_translate(args):
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        raise SystemExit("set YANDEX_API_KEY and YANDEX_FOLDER_ID")

    with open(args.inp, encoding="utf-8") as f:
        src_lines = [line.rstrip("\n") for line in f]

    to_translate = [(i, s) for i, s in enumerate(src_lines) if s.strip()]
    out_lines = [""] * len(src_lines)

    batches = chunk_lines(to_translate, args.batch_texts, args.batch_chars)
    start = perf_counter()
    total = len(to_translate)
    done = 0

    pbar = tqdm(total=total, unit="lines") if tqdm else None

    for bi, batch in enumerate(batches, 1):
        idxs = [i for i, _ in batch]
        texts = [t for _, t in batch]
        translated = yandex_batch(texts, api_key, folder_id, args.source, args.target)

        for i, t in zip(idxs, translated):
            out_lines[i] = fix_script(t)

        done += len(batch)
        elapsed = perf_counter() - start
        rate = done / max(1e-9, elapsed)

        if pbar:
            pbar.update(len(batch))
            pbar.set_postfix_str(f"batch {bi}/{len(batches)} | {rate:.1f}/s")
        else:
            eta = (total - done) / max(1e-9, rate)
            print(f"{done}/{total} ({100 * done / total:.1f}%) | {rate:.1f}/s | ETA {eta:.0f}s", end="\r")

    if pbar:
        pbar.close()
    else:
        print()

    with open(args.out, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"wrote {args.out} ({len(out_lines)} lines)")


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", (text or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json found")
    return json.loads(m.group(0))


def openrouter_fix(api_key, model, en, cv):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "chuvash-txt-fixer",
    }
    system = (
        "You are a careful translator into Chuvash written in Cyrillic.\n"
        "Given an English source sentence and a candidate Chuvash translation, output a corrected "
        "Chuvash (Cyrillic) translation.\n"
        "Requirements:\n"
        "- Output MUST be in Chuvash (Cyrillic), not Russian.\n"
        "- Preserve meaning, names, numbers, punctuation.\n"
        "- If the candidate has Latin diacritics like ă ĕ ç ÿ, convert them to ӑ ӗ ҫ ӳ.\n"
        "Return STRICT JSON ONLY with exactly these keys:\n"
        '  {"fixed_translation": string, "issues": string[]}\n'
        "No extra keys. No commentary outside JSON."
    )
    payload = {
        "model": model, "temperature": 0, "max_tokens": 600,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"English source:\n{en}\n\nCandidate translation:\n{cv}"},
        ],
    }

    for attempt in range(1, 4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return parse_json(r.json()["choices"][0]["message"]["content"])
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * attempt)


def run_fix(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY")

    en_lines = open(args.en, encoding="utf-8").read().splitlines()
    cv_lines = open(args.cv, encoding="utf-8").read().splitlines()

    n = min(len(en_lines), len(cv_lines))
    if args.max_rows > 0:
        n = min(n, args.max_rows)

    en_lines, cv_lines = en_lines[:n], cv_lines[:n]
    fixed_lines = [""] * n
    report = []

    it = tqdm(range(n), desc="fixing") if tqdm else range(n)
    calls = 0

    for i in it:
        src = norm(en_lines[i])
        cand_orig = cv_lines[i]
        cand = fix_script(cand_orig)
        called = False
        issues = []
        fixed = cand

        if src.strip() and not has_chuv(cand):
            called = True
            obj = openrouter_fix(api_key, args.model, src, cand)
            fixed = fix_script(str(obj.get("fixed_translation", "")))
            raw = obj.get("issues", [])
            issues = [str(x) for x in raw] if isinstance(raw, list) else [str(raw)]
            calls += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        fixed_lines[i] = fixed
        report.append({
            "line": i, "called": called,
            "chuv_before": has_chuv(cand), "chuv_after": has_chuv(fixed),
            "en": src, "cv_orig": cand_orig, "cv_norm": cand, "cv_fixed": fixed,
            "issues": " | ".join(issues),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for line in fixed_lines:
            f.write(line + "\n")

    pd.DataFrame(report).to_csv(args.report, index=False)
    print(f"wrote {args.out} ({n} lines)")
    print(f"wrote {args.report}")
    print(f"openrouter calls: {calls}/{n}")


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)

p1 = sp.add_parser("select")
p1.add_argument("--test-csv", default="test.csv")
p1.add_argument("--out", default="")
p1.add_argument("--n-neighbors", type=int, default=20)
p1.add_argument("--cover-per-test", type=int, default=8)
p1.add_argument("--total-k", type=int, default=8000)
p1.add_argument("--sim-floor", type=float, default=0.18)
p1.add_argument("--min-chars", type=int, default=50)
p1.add_argument("--max-chars", type=int, default=260)
p1.add_argument("--min-words", type=int, default=9)

p2 = sp.add_parser("translate")
p2.add_argument("--in", dest="inp", default="english.txt")
p2.add_argument("--out", default="chuvash.txt")
p2.add_argument("--source", default="en")
p2.add_argument("--target", default="cv")
p2.add_argument("--batch-texts", type=int, default=50)
p2.add_argument("--batch-chars", type=int, default=9000)

p3 = sp.add_parser("fix")
p3.add_argument("--en", default="english.txt")
p3.add_argument("--cv", default="chuvash.txt")
p3.add_argument("--out", default="fixed_chuvash.txt")
p3.add_argument("--report", default="fix_report.csv")
p3.add_argument("--model", default="google/gemini-2.5-flash")
p3.add_argument("--sleep", type=float, default=0.0)
p3.add_argument("--max-rows", type=int, default=0)

args = ap.parse_args()
{"select": run_select, "translate": run_translate, "fix": run_fix}[args.cmd](args)
