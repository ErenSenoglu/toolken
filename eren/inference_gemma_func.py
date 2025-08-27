import os
import json
import time
import re
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext

import fire
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from funchub.math import *  # noqa: F401,F403 - used by eval("<op>_...") as in LLaMA path
from evaluation.metrics import parse_answer, accuracy

# Soften/disable Torch Dynamo during inference to avoid recompilation limit hits
try:
    import torch._dynamo as _dynamo  # type: ignore
    _DYNAMO_DISABLE = getattr(_dynamo, "disable", None) or nullcontext
    # Bump limits if Dynamo is active
    try:
        _dynamo.config.cache_size_limit = max(getattr(_dynamo.config, "cache_size_limit", 8), 64)
        _dynamo.config.recompile_limit = max(getattr(_dynamo.config, "recompile_limit", 8), 64)
    except Exception:
        pass
except Exception:
    _DYNAMO_DISABLE = nullcontext


def load_gemma(model_name_or_path: str, device: torch.device, dtype: str = "bf16"):
    if torch.cuda.is_available():
        if dtype.lower() == "bf16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif dtype.lower() == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map=None)
    model.to(device)
    model.eval()
    return model, tok


class FunctionHeadOnlyLight(nn.Module):
    """Minimal function head wrapper: Linear(H -> K) with id2func mapping."""

    def __init__(self, base_model, func_dict: Dict[str, int]):
        super().__init__()
        self.model = base_model
        self.func_dict = func_dict
        self.id2func = {v: k for k, v in func_dict.items()}
        cfg = self.model.config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "n_embd", None)
        )
        assert hidden_size is not None, "Could not infer hidden size from model config"
        self.func_head = nn.Linear(int(hidden_size), len(func_dict), bias=False)

    def load_weights(self, pth_path: str):
        state = torch.load(pth_path, map_location="cpu")
        # Accept cases where state may be under 'state_dict' or direct
        if isinstance(state, dict) and "func_head.weight" in state:
            self.load_state_dict(state, strict=False)
        else:
            self.func_head.load_state_dict(state, strict=True)
        self.func_head.to(next(self.model.parameters()).device)
        self.eval()

    @torch.inference_mode()
    def last_step_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]  # [B,H]
        logits = self.func_head(h.float())   # [B,K]
        return torch.softmax(logits, dim=-1)


def read_templates(dataset: str) -> Dict[str, str]:
    tdir = f"data/{dataset}/template"
    templates: Dict[str, str] = {}
    for name in os.listdir(tdir):
        with open(os.path.join(tdir, name), "r", encoding="utf-8") as f:
            templates[name.split("_")[-1].replace(".txt", "")] = f.read()
    return templates


def load_func_dict(dataset: str) -> Dict[str, int]:
    path = f"data/{dataset}/func_dict.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_questions(dataset: str) -> Tuple[List[str], List[float]]:
    test_path = f"data/{dataset}/test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    questions: List[str] = []
    labels: List[float] = []
    for item in data:
        q = item["question"]
        vlist = item["enhanced_v"]
        for i, val in enumerate(vlist):
            q = q.replace(f"{{v_{i+1}}}", str(val))
        questions.append(q)
        labels.append(float(item.get("enhanced_result", item.get("result", 0.0))))
    return questions, labels


class DigitsMaskOnce(LogitsProcessor):
    def __init__(self, tokenizer, enable: bool = False):
        self.tokenizer = tokenizer
        self.enable = enable
        # Precompute token ids that decode to single digits 0-9
        self.digit_ids = set()
        for tid in range(self.tokenizer.vocab_size):
            try:
                txt = self.tokenizer.decode([tid], skip_special_tokens=True)
            except Exception:
                continue
            if len(txt) == 1 and txt.isdigit():
                self.digit_ids.add(tid)

    def __call__(self, input_ids, scores):
        if not self.enable:
            return scores
        scores[:, list(self.digit_ids)] = -1e9
        # one-shot mask; disable after use
        self.enable = False
        return scores


class AnswerStopper(StoppingCriteria):
    pattern = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?(?=\s+[^\d,\.]|\n)")

    def __init__(self, tok, start_idx: int):
        super().__init__()
        self.tok = tok
        self.start_idx = start_idx

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:  # type: ignore[override]
        seq = input_ids[0]
        if seq.shape[0] <= self.start_idx:
            return False
        gen_part = seq[self.start_idx:]
        tail_ids = gen_part[-128:]
        text_tail = self.tok.decode(tail_ids, skip_special_tokens=True)
        return bool(self.pattern.search(text_tail))


def parse_and_eval(op: str, args_raw: str) -> Tuple[bool, str, float]:
    # Cleanup similar to LLaMA path
    s = args_raw.replace("$", "")
    if ", " in s:
        s = s.replace(", ", ";").replace(",", "").replace(";", ", ")
    s = s.replace(" ", "")
    if "(" not in s or ")" not in s:
        return False, args_raw, 0.0
    if "%" in s:
        temp = s.split("(")[1].split(")")[0].split(",")
        for i, a in enumerate(temp):
            if "%" in a:
                a = a.replace("%", "").strip()
                a = str(float(a) / 100)
            temp[i] = a
        s = f"({', '.join(temp)})"
    try:
        res = eval(f"{op[1:-1]}_{s}")
        return True, s, float(res)
    except Exception:
        return False, s, 0.0


@torch.inference_mode()
def func_infer_one(
    model,
    tok,
    templates: Dict[str, str],
    question: str,
    max_gen_len: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    head: FunctionHeadOnlyLight | None = None,
    max_func_calls: int = 16,
    op_arg_retry_limit: int = 1,
):
    func_map = ["<add>", "<subtract>", "<multiply>", "<divide>"]
    cur_gen = ""
    cur_gen_with_func = ""
    func_calls: List[str] = []
    start_len: List[int] = []
    end_len: List[int] = []

    digits_mask = DigitsMaskOnce(tok)
    # Track last appended token text to clean up stray single-digit before function start
    last_token_text: str = ""
    last_was_token: bool = False
    # Track function call counts and pending failures for arg completion
    func_calls_made: int = 0
    pending_op: Optional[str] = None
    pending_failures: int = 0

    # Fast path for baseline (no head): single-pass with general template
    if head is None:
        base_tmpl = templates.get("general", next(iter(templates.values())))
        prompt = base_tmpl.replace("[QUESTION]", question)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        stopper = StoppingCriteriaList([AnswerStopper(tok, prompt_len)])
        out = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-6) if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=stopper,
        )
        cur_gen = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return {
            "func_calls": [],
            "generation": cur_gen.strip(),
            "status": "success",
        }

    # With a trained head: decode step-by-step using JOINT space (tokens + function classes)
    # Helper to sample from concatenated logits [V+K]
    def sample_joint(token_logits: torch.Tensor, func_logits: torch.Tensor):
        # token_logits: [1,V], func_logits: [1,K]
        V = token_logits.size(-1)
        K = func_logits.size(-1)
        joint = torch.cat([token_logits, func_logits], dim=-1)  # [1,V+K]
        if temperature and temperature > 0.0:
            joint = joint / max(temperature, 1e-6)
            probs = torch.softmax(joint, dim=-1)[0]
            if top_p and 0.0 < top_p < 1.0:
                # Nucleus sampling over joint
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > float(top_p)
                # keep at least one
                cutoff_idx = int(torch.nonzero(cutoff, as_tuple=False)[0].item()) if torch.any(cutoff) else sorted_probs.numel()
                keep = sorted_idx[:cutoff_idx + 1]
                probs_masked = torch.zeros_like(probs)
                probs_masked[keep] = probs[keep]
                probs = probs_masked / probs_masked.sum()
            choice = int(torch.multinomial(probs, num_samples=1).item())
        else:
            choice = int(torch.argmax(joint[0]).item())
        if choice < V:
            return ("token", choice)
        else:
            return ("func", choice - V)

    for _ in range(max_gen_len):
        # Main decoding loop always uses the general template
        base_tmpl = templates.get("general", next(iter(templates.values())))
        prompt = base_tmpl.replace("[QUESTION]", question) + cur_gen
        enc = tok(prompt, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        # Forward to get logits and last hidden for head
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        token_logits_last = outputs.logits[:, -1, :].float()  # [1,V]

        # Optional one-shot mask to avoid immediate digit append after numeric replacement
        if digits_mask.enable:
            if getattr(digits_mask, "digit_ids", None):
                dids = list(digits_mask.digit_ids)
                token_logits_last[:, dids] = -1e9
            digits_mask.enable = False

        # Function head logits
        last_h = outputs.hidden_states[-1][:, -1, :]
        func_logits_last = head.func_head(last_h.float())  # [1,K]

        # Joint selection
        kind, val = sample_joint(token_logits_last, func_logits_last)
        if kind == "token":
            new_text = tok.decode([val], skip_special_tokens=True)
            if new_text == "":
                new_text = tok.decode([val], skip_special_tokens=False)
            cur_gen += new_text
            last_token_text = new_text
            last_was_token = True
        else:
            # Function selected: append op start and immediately go to arg completion path
            func_idx = val
            op = head.id2func.get(int(func_idx), None)
            if op is None:
                # fallback: skip if unknown
                continue
            # If last appended token was a single-digit (possibly with leading space), remove it to avoid prefixing result
            if last_was_token and re.fullmatch(r"\s?\d", last_token_text or ""):
                cur_gen = cur_gen[:-len(last_token_text)]
            last_token_text = ""
            last_was_token = False
            cur_gen += f"{op}("
            pending_op = op
            pending_failures = 0

        # If the decoded text ends with an op-start, switch to argument completion
        for op in func_map:
            if cur_gen.endswith(op + "("):
                # Build cur_gen_with_func by inserting previous evaluated calls
                if start_len and end_len:
                    tmp = cur_gen
                    bias = 0
                    for i in range(len(start_len)):
                        tmp = tmp[: start_len[i] + bias] + func_calls[i] + tmp[end_len[i] + bias :]
                        bias += len(func_calls[i]) - (end_len[i] - start_len[i])
                    cur_gen_with_func = tmp
                else:
                    cur_gen_with_func = cur_gen

                # For sub-generation (completing args), prefer op-specific template; fallback to the func template, else general
                fallback_tmpl = templates.get("func", base_tmpl)
                sub_prompt = templates.get(op[1:-1], fallback_tmpl).replace("[QUESTION]", question) + cur_gen_with_func
                sub_inputs = tok(sub_prompt, return_tensors="pt").to(model.device)
                sub_len = sub_inputs["input_ids"].shape[-1]
                sub_out = model.generate(
                    **sub_inputs,
                    max_new_tokens=48,
                    do_sample=(temperature > 0.0),
                    temperature=max(temperature, 1e-6) if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
                sub_text = tok.decode(sub_out[0][sub_len:], skip_special_tokens=True)
                # Extract args up to the first ')'
                after = cur_gen + sub_text
                tail = after.split(op, 1)[-1]
                args_part = tail.split(")", 1)[0] + ")"

                ok, args_norm, res = parse_and_eval(op, args_part)
                if ok:
                    func_calls.append(f"{op}{args_norm} = {res}")
                    start_len.append(len(cur_gen.split(op)[0]))
                    cur_gen = cur_gen.split(op)[0] + str(res)
                    end_len.append(len(cur_gen))
                    # Next token: avoid immediate digit repetition
                    digits_mask.enable = True
                    last_token_text = ""
                    last_was_token = False
                    func_calls_made += 1
                    pending_op = None
                    pending_failures = 0
                    if func_calls_made >= max_func_calls:
                        # Reached maximum allowed function calls; stop outer loop
                        return {
                            "func_calls": func_calls,
                            "generation": cur_gen.strip(),
                            "status": "max_func_calls_reached",
                        }
                else:
                    # Failed to parse/eval; retry up to limit, then remove dangling op(
                    if pending_op == op:
                        pending_failures += 1
                        if pending_failures >= max(0, int(op_arg_retry_limit)):
                            # Back off: drop the dangling op( and continue normal decoding
                            cur_gen = cur_gen.rsplit(op + "(", 1)[0]
                            pending_op = None
                            pending_failures = 0
                break

        # Early stop if #### <number> emitted (require a delimiter after full number)
        if re.search(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?(?=\s+[^\d,\.]|\n)", cur_gen):
            break

    return {
        "func_calls": func_calls,
        "generation": cur_gen.strip(),
        "status": "success",
    }


def main(
    model_name_or_path: str = "google/gemma-3-12b-pt",
    dataset: str = "gsm8k-xl",
    max_gen_len: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_samples: Optional[int] = None,
    dtype: str = "bf16",
    output_dir: str = "outputs/gsm8k-xl",
    output_name: str = "inference-gemma-func-12b.jsonl",
    func_head_path: Optional[str] = None,
    max_func_calls: int = 16,
    op_arg_retry_limit: int = 1,
    # Resume support: start processing from this dataset index and append to existing JSONL
    continue_index: Optional[int] = None,
):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok = load_gemma(model_name_or_path, device, dtype)
    head = None
    if func_head_path:
        func_dict = load_func_dict(dataset)
        head = FunctionHeadOnlyLight(model, func_dict)
        head.load_weights(func_head_path)
    templates = read_templates(dataset)
    questions, labels = build_questions(dataset)
    if max_samples is not None:
        questions = questions[:max_samples]
        labels = labels[:max_samples]

    # Prepare resume/append
    preds: List[float] = []
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)

    existing_indices: set[int] = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        rec = json.loads(line)
                        idx = rec.get("index")
                        if isinstance(idx, int):
                            existing_indices.add(idx)
                    except Exception:
                        continue
        except Exception:
            pass

    # Determine starting index
    if continue_index is not None:
        start_idx = max(0, int(continue_index))
    else:
        start_idx = (max(existing_indices) + 1) if existing_indices else 0

    # Clamp to available data
    start_idx = min(start_idx, len(questions))

    # Compute how many to run (skip already-present indices as safety)
    remaining_indices = [i for i in range(start_idx, len(questions)) if i not in existing_indices]
    if continue_index is not None:
        print(f"[RESUME] continue_index={continue_index}; existing records in {output_name}: {len(existing_indices)}; will run {len(remaining_indices)} items (from {start_idx}).")
    elif existing_indices:
        print(f"[RESUME] Auto-resume detected; existing records in {output_name}: {len(existing_indices)}; starting at index {start_idx}.")
    else:
        print("[START] No existing outputs; starting from index 0.")

    # Append to file; create if missing
    with open(out_path, "a", encoding="utf-8") as f:
        for i in tqdm(remaining_indices, total=len(remaining_indices), desc="FuncInfer"):
            q, gold = questions[i], labels[i]
            log = func_infer_one(
                model,
                tok,
                templates,
                q,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                head=head,
                max_func_calls=max_func_calls,
                op_arg_retry_limit=op_arg_retry_limit,
            )
            gen = log["generation"]
            pv = parse_answer(gen, pattern="####")
            preds.append(pv)
            rec = {
                "index": i,
                "question": q,
                "generation": gen,
                "pred": pv,
                "gold": gold,
                "func_calls": log.get("func_calls", []),
                "status": log.get("status", "success"),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

    # Evaluate only on newly processed items
    dur = time.time() - t0
    if preds:
        new_labels = [labels[i] for i in remaining_indices]
        em = accuracy(preds, new_labels, type="em")
        approx = accuracy(preds, new_labels, type="approx")
        print(f"[DONE] Saved outputs to {out_path} | EM(new)={em:.4f} | approx(new)={approx:.4f} | n_new={len(preds)} | time={dur:.1f}s")
    else:
        print(f"[DONE] Saved outputs to {out_path} | No new items processed | time={dur:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)