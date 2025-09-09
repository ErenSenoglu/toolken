import os
import json
import random
from typing import Dict, List, Tuple

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional Weights & Biases logging
try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# =============================
# Joint-concat trainer for Gemma
# - Freeze Gemma base LM
# - Train a small linear head (hidden_size -> K functions)
# - Concatenate token logits (V) and function logits (K) for next-token prediction
# =============================


def set_seeds(seed: int = 1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def find_sublist(haystack: List[int], needle: List[int], start_from: int = 0) -> int:
    """Return first index where needle appears in haystack, else -1."""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(start_from, n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


def load_gemma_model(model_name_or_path: str, device: torch.device, dtype: torch.dtype | None = None):
    """Load Gemma model+tokenizer; keep it simple and place on the given device."""
    if dtype is None:
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    return model, tokenizer


class FunctionHeadOnly(nn.Module):
    """Wrap a frozen LM + a trainable linear function head.

    - Hidden states: [B, T, H]
    - Function head: Linear(H -> K)
    - Loss at function-start positions only (labels in [0..K-1]); others masked (-100)
    """

    def __init__(self, base_model, tokenizer, func_dict: Dict[str, int]):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.func_dict = func_dict
        self.id2func = {v: k for k, v in func_dict.items()}
        self._printed_func_range = False

        # Infer hidden size from config (Gemma 3 text hidden size = 2560 for 4B)
        cfg = self.model.config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "n_embd", None)
        )
        assert hidden_size is not None, "Could not infer hidden size from model config"

        # Small linear head: H -> K (no bias, to mirror typical tied-head setups)
        self.func_head = nn.Linear(int(hidden_size), len(func_dict), bias=False)

        # Freeze the base LM; we only train func_head
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _encode_with_specials(self, text: str) -> List[int]:
        ids = self.tokenizer(text, add_special_tokens=True, return_tensors=None)["input_ids"]
        return ids[0] if ids and isinstance(ids[0], list) else ids

    def _encode_no_specials(self, text: str) -> List[int]:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
        return ids[0] if ids and isinstance(ids[0], list) else ids

    @property
    def base_vocab_size(self) -> int:
        # Gemma 3 4B-PT text vocab is 256000 (0..255999); read from config/tokenizer for safety
        return int(getattr(self.model.config, "vocab_size", self.tokenizer.vocab_size))

    def build_labels(self, example: dict, text_ids: List[int], debug: bool = False) -> Tuple[List[int], List[dict]]:
        """Create head-only labels for one example.

        - Returns labels: length == len(text_ids)
        values: func_id in [0..K-1] at function-start positions, else -100
        - Also returns a debug list with placement info.
        """
        labels = [-100] * len(text_ids)
        ops_dbg: List[dict] = []

        # Operators to look for (preferred via tar_eq)
        tar_eqs = example.get("tar_eq")
        if not tar_eqs:
            tar_eqs = ["<" + example["api"] + ">"] if "api" in example else []

        # Prefer dataset-provided token spans; try direct (with specials) first, else compute offset from no-specials
        start_idxs = example.get("start_token_idx")
        end_idxs = example.get("end_token_idx")
        offset = -1
        if isinstance(start_idxs, list) and isinstance(end_idxs, list) and len(start_idxs) == len(end_idxs) == len(tar_eqs):
            # Precompute offset for fallback only
            base_ids = self._encode_no_specials(example["text"])
            offset = find_sublist(text_ids, base_ids, start_from=0)

        search_from = 0
        for j, eq in enumerate(tar_eqs):
            # Extract operator token like "<add>" or "[api]"
            op = None
            if "[" in eq:
                import re as _re
                m = _re.search(r"(\[.*?\])", eq)
                if m:
                    op = m.group(1)
            if op is None and "<" in eq:
                import re as _re
                m = _re.search(r"(<.*?>)", eq)
                if m:
                    op = m.group(1)
            if op is None:
                continue
            if op not in self.func_dict:
                op = op[1:-1]  # fallback strip brackets
            if op not in self.func_dict:
                ops_dbg.append({"raw_eq": eq, "op": op, "pos": -1, "via": "unknown-op"})
                continue

            if isinstance(start_idxs, list) and isinstance(end_idxs, list) and j < len(start_idxs):
                #MODIFICATIONS HERE!! LLama and Phi
                s_direct = int(start_idxs[j]) 
                e_direct = int(end_idxs[j])  if j < len(end_idxs) else s_direct
                if 0 <= s_direct < len(labels):
                    labels[s_direct] = int(self.func_dict[op])
                    ops_dbg.append({"raw_eq": eq, "op": op, "pos": s_direct, "len": max(0, e_direct - s_direct), "via": "idx-direct"})
                    continue
                elif offset != -1:
                    s = s_direct + int(offset)
                    e = e_direct + int(offset)
                    if 0 <= s < len(labels):
                        labels[s] = int(self.func_dict[op])
                        ops_dbg.append({"raw_eq": eq, "op": op, "pos": s, "len": max(0, e - s), "via": "idx-offset"})
                        continue
                    else:
                        ops_dbg.append({"raw_eq": eq, "op": op, "pos": -1, "len": 0, "via": "idx-offset-oob"})
                        # Fallthrough to search

            op_ids = self._encode_no_specials(op)
            s = find_sublist(text_ids, op_ids, start_from=search_from)
            if s == -1:
                s = find_sublist(text_ids, op_ids, start_from=0)
            if s == -1:
                ops_dbg.append({"raw_eq": eq, "op": op, "pos": -1, "len": len(op_ids), "via": "search"})
                continue
            labels[s] = int(self.func_dict[op])
            search_from = s + len(op_ids)
            ops_dbg.append({"raw_eq": eq, "op": op, "pos": s, "len": len(op_ids), "via": "search"})

        if debug:
            # Print a small context window around each placed label
            toks = text_ids
            for o in ops_dbg:
                pos = int(o.get("pos", -1))
                if pos >= 0:
                    lo = max(0, pos - 5)
                    hi = min(len(toks), pos + 6)
                    print(f"[DBG] op={o['op']} via={o['via']} pos={pos} span_len={o['len']} tokens[{lo}:{hi}]={toks[lo:hi]}")

        return labels, ops_dbg

    def loss_on_example(
        self,
        example: dict,
        only_functoken: bool = False,
        inspect: bool = False,
        inspect_topk: int = 5,
        inspect_limit: int = 5,
    ):
        """Compute joint-concat loss and per-class counts; optionally mine gating supervision.

        Miner logic (teacher-forced pass) summary:
        For each time step i (prefix x_{<=i}):
            - tool distribution = softmax over function head logits at position i
            - gold_is_tool = whether next token is a tool-start (function label at that position)
            - Case A (REJ): gold_is_tool == False AND (top1_tool_prob >= alpha) -> record sample labeled 'REJ'
            - Case B (tool disambiguation): gold_is_tool == True -> record sample labeled with gold tool name; ensure gold in candidates (force-include if needed)
        Persist JSON lines with: prefix_text, candidates (tool names), gold_label, margins, entropy, probs.
        """
        device = next(self.parameters()).device

        # Tokenize text
        text_ids = self._encode_with_specials(example["text"])  # includes specials
        input_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # Build function labels at token positions (start positions only)
        labels_list, _ = self.build_labels(example, text_ids, debug=False)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device).unsqueeze(0)

        # Forward frozen LM to get last hidden state and token logits
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [1, T, H]
            token_logits_full = outputs.logits  # [1, T, V]

        # Function logits in float32 for numerical stability
        func_logits_full = self.func_head(hidden.float())  # [1, T, K]

        # Use the actual token logits dimension (can differ from config.vocab_size after resizing)
        V = int(token_logits_full.size(-1))
        K = len(self.func_dict)

        # Use positions 0..T-2 to predict next token
        token_logits = token_logits_full[:, :-1, :].float()  # [1, T-1, V]
        func_logits = func_logits_full[:, :-1, :].float()    # [1, T-1, K]
        comb_logits = torch.cat([token_logits, func_logits], dim=-1)  # [1, T-1, V+K]

        next_tokens = input_ids[:, 1:]  # [1, T-1]
        joint_targets = next_tokens.clone()

        # Overwrite targets at function-start positions with V + func_id
        # OFF-BY-ONE FIX:
        # We predict next_tokens[t] using prefix ending at position t. A tool-start
        # event corresponds to the NEXT token being a tool-start token. The raw
        # labels[] array marks the position of the tool-start token itself. Therefore
        # we must look at labels[:, 1:] (the next token) when deciding whether the
        # target at time step t should be remapped into the tool class range.
        func_labels_next = labels[:, 1:]          # align with next_tokens
        func_mask = func_labels_next != -100
        if func_mask.any():
            joint_targets[func_mask] = V + func_labels_next[func_mask]

        # Optional inspection: print minimal joint top-k at function positions
        if inspect:
            if not self._printed_func_range:
                print(f"[INFO] Function classes occupy indices [{V}..{V+K-1}] in the concatenated logits (V={V}, K={K}).")
                self._printed_func_range = True
            with torch.no_grad():
                # Flatten function mask to positions within 0..T-2
                func_pos = torch.nonzero(func_mask.view(-1), as_tuple=False).view(-1).tolist()
                shown = 0
                for fp in func_pos:
                    if shown >= max(0, int(inspect_limit)):
                        break
                    tpos = fp  # time-step index in 0..T-2
                    # Joint-space probabilities over [0..V+K-1]
                    joint_probs = F.softmax(comb_logits.view(-1, V + K)[tpos].float(), dim=-1)
                    j_topk = min(max(1, int(inspect_topk)), V + K)
                    j_p, j_id = torch.topk(joint_probs, k=j_topk)
                    joint_pairs = []
                    for i in range(j_p.numel()):
                        cid = int(j_id[i].item())
                        if cid < V:
                            name = self.tokenizer.decode([cid])
                            disp = name
                        else:
                            fid = cid - V
                            name = self.id2func.get(fid, str(fid))
                            disp = f"FUNC:{name}"
                        joint_pairs.append((disp, cid, float(j_p[i].item())))

                    print(f"[JOINT] t={tpos} topk:", [(cid, disp, round(prob, 4)) for (disp, cid, prob) in joint_pairs])
                    shown += 1

        flat_logits = comb_logits.reshape(-1, V + K)
        flat_targets = joint_targets.view(-1)

        if only_functoken:
            valid = flat_targets >= V
            if valid.any():
                loss = F.cross_entropy(flat_logits[valid], flat_targets[valid], reduction="mean")
            else:
                loss = flat_logits.sum() * 0.0
        else:
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="mean")

        # Per-class counts over function positions
        with torch.no_grad():
            preds = torch.argmax(flat_logits, dim=-1)
            fmask = flat_targets >= V
            f_true = flat_targets[fmask] - V
            f_pred = preds[fmask] - V
            if K == 0 or f_true.numel() == 0:
                results = {"tp": np.array([]), "pred": np.array([]), "true": np.array([])}
            else:
                tp = torch.zeros((K,), dtype=torch.long, device=flat_logits.device)
                pc = torch.zeros((K,), dtype=torch.long, device=flat_logits.device)
                tc = torch.zeros((K,), dtype=torch.long, device=flat_logits.device)
                for c in range(K):
                    cm_t = f_true == c
                    cm_p = f_pred == c
                    tp[c] = torch.sum(cm_t & cm_p)
                    pc[c] = torch.sum(cm_p)
                    tc[c] = torch.sum(cm_t)
                results = {"tp": tp.cpu().numpy(), "pred": pc.cpu().numpy(), "true": tc.cpu().numpy()}

        return loss, results


def aggregate_metrics(batches: List[Dict[str, np.ndarray]]):
    """Sum tp/pred/true across a list of metrics dicts and compute micro P/R/F1.

    Returns dict with micro metrics; classwise arrays can be computed via sum_class_counters.
    """
    if not batches:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(np.sum(b["tp"]) for b in batches if b["tp"].size)
    pred = sum(np.sum(b["pred"]) for b in batches if b["pred"].size)
    true = sum(np.sum(b["true"]) for b in batches if b["true"].size)
    precision = (tp / pred) if pred > 0 else 0.0
    recall = (tp / true) if true > 0 else 0.0
    f1 = (2 * tp / (pred + true)) if (pred + true) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def sum_class_counters(batches: List[Dict[str, np.ndarray]], K: int):
    """Aggregate per-class tp/pred/true arrays across batches."""
    tp = np.zeros((K,), dtype=np.int64)
    pred = np.zeros((K,), dtype=np.int64)
    true = np.zeros((K,), dtype=np.int64)
    for b in batches:
        if b["tp"].size:
            tp += b["tp"].astype(np.int64)
            pred += b["pred"].astype(np.int64)
            true += b["true"].astype(np.int64)
    return tp, pred, true


def per_class_metrics(tp: np.ndarray, pred: np.ndarray, true: np.ndarray):
    """Compute per-class precision/recall/F1 arrays and macro averages."""
    eps = 1e-12
    prec = np.divide(tp, np.maximum(pred, 1), dtype=np.float64)
    rec = np.divide(tp, np.maximum(true, 1), dtype=np.float64)
    denom = pred + true
    f1 = np.divide(2 * tp, np.maximum(denom, 1), dtype=np.float64)
    return prec, rec, f1, float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1))


def main(
    model_name_or_path: str = "/workspace/.hf_home/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    input_file: str = "data/funcqa/train.json",
    dataset: str = "funcqa",  # used to pick function dictionary
    lr: float = 1e-3,
    num_epochs: int = 3,  # Qwen use 5, Phi use 7
    max_train_samples: int | None = 1024,  # keep small for overfit/debug
    shuffle_each_epoch: bool = True,
    save_dir: str = "checkpoints_plus",
    use_wandb: bool = True,
    wandb_project: str = "toolkengpt-funcqa",
    wandb_run_name: str | None = None,
    only_functoken: bool = False,
    inspect_token_logits: bool = False,
    inspect_topk: int = 5,
    inspect_limit: int = 3,
    miner_enabled: bool = False,              # if True: after training, mine validation set using best checkpoint
    post_mine_top_k: int = 3,
    post_mine_output: str = "miner_small_split.jsonl",
    mining_only: bool = True,                # NEW: skip training and run only mining phase
    load_checkpoint: str | None = "/workspace/toolken/emre_code/checkpoints_head_only/funcqa/google/gemma-3-4b-pt/head_best.pth",       # NEW: path to existing func_head weights (.pth) to use (overrides best)
    append_train_raw: bool = False,           # NEW: after mining val, append raw train examples (no per-position expansion)
):
    """Train with joint logits concatenation (token + function) like train_llama."""
    set_seeds(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = save_dir + f"/{model_name_or_path}"

    # 1) Load function classes
    func_dict_path = f"./data/{dataset}/func_dict.json"
    assert os.path.exists(func_dict_path), f"func_dict not found: {func_dict_path}"
    with open(func_dict_path, "r", encoding="utf-8") as f:
        func_dict = json.load(f)
    K = len(func_dict)
    print(f"[INFO] Loaded function dict with {K} classes from {func_dict_path}")

    # 2) Load model+tokenizer
    model, tokenizer = load_gemma_model(model_name_or_path, device=device)
    text_hidden = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd", None)
        or getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    )
    print(f"[INFO] Model dtype={model.dtype}, hidden_size={text_hidden}, vocab_size={getattr(model.config, 'vocab_size', tokenizer.vocab_size)}")

    # 3) Wrap with our head-only module
    head_model = FunctionHeadOnly(model, tokenizer, func_dict).to(device)
    optimizer = torch.optim.Adam([p for p in head_model.parameters() if p.requires_grad], lr=lr)

    # Optional: initialize Weights & Biases (disabled automatically if mining_only)
    if mining_only:
        if use_wandb:
            print("[INFO] mining_only=True -> disabling wandb init for this run.")
        use_wandb = False
    if use_wandb:
        if wandb is None:
            print("[WARN] wandb not installed; set use_wandb=False or `pip install wandb` to enable logging.")
        else:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"gemma3-{dataset}",
                config={
                    "model": model_name_or_path,
                    "dataset": dataset,
                    "K": K,
                    "lr": lr,
                    "num_epochs": num_epochs,
                    "max_train_samples": max_train_samples,
                    "mining_only": mining_only,
                },
            )

    # 4) Load dataset
    assert os.path.exists(input_file), f"input_file not found: {input_file}"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f) if input_file.endswith(".json") else [json.loads(line) for line in f]
    print(f"[INFO] Loaded {len(data)} examples from {input_file}")

    # Keep a tiny subset for debug/overfit
    if max_train_samples is not None:
        data = data[: max_train_samples]
        print(f"[INFO] Using first {len(data)} examples for training (max_train_samples)")

    import random
    SEED = 42
    n = len(data)
    val_fraction = 0.1
    min_val_samples = 16
    # Compute validation length based on user-specified fraction & minimum, allowing 0 to disable val set.
    if val_fraction <= 0.0 or n == 0:
        val_len = 0
    else:
        val_len = min(max(min_val_samples, int(val_fraction * n)), n)
    random.seed(SEED)
    all_idx = list(range(n))
    if val_len > 0:
        val_idx = set(random.sample(all_idx, val_len))
        val_data = [data[i] for i in val_idx]
        train_data = [data[i] for i in all_idx if i not in val_idx]
        print(sorted(list(val_idx))[:20])  # preview indices
    else:
        val_idx = set()
        val_data = []
        train_data = data
    print(f"[INFO] val_fraction={val_fraction} min_val_samples={min_val_samples} -> val_len={val_len}")
    print(f"[INFO] Train size={len(train_data)} | Val size={len(val_data)} (seed={SEED})")

    os.makedirs(save_dir, exist_ok=True)    # Quick sanity pass: count how many examples contain at least one supervised position


    def count_supervised(examples: List[dict], max_check: int = 200) -> int:
        cnt = 0
        for ex in examples[:max_check]:
            text_ids = head_model._encode_with_specials(ex["text"]) if isinstance(ex.get("text"), str) else []
            if text_ids:
                labels, _ = head_model.build_labels(ex, text_ids, debug=False)
                if any(l != -100 for l in labels):
                    cnt += 1
        return cnt

    sup_train = count_supervised(train_data)
    sup_val = count_supervised(val_data)
    print(f"[INFO] Supervised examples (first pass): train={sup_train}/{len(train_data)}, val={sup_val}/{len(val_data)}")

    # Training loop (NOTE: in-training hard mining removed; we now mine only AFTER training with best checkpoint)
    global_step = 0
    best_val_f1 = -1.0
    best_path = None

    if mining_only:
        print("[INFO] mining_only=True -> Skipping training loop.")
        # If user supplies a checkpoint, load it here (func_head only)
        if load_checkpoint and os.path.exists(load_checkpoint):
            head_model.func_head.load_state_dict(torch.load(load_checkpoint, map_location=device))
            best_path = load_checkpoint
            print(f"[INFO] Loaded provided checkpoint: {load_checkpoint}")
        else:
            # Try default best path under save_dir
            default_best = os.path.join(save_dir, "head_best.pth")
            if os.path.exists(default_best):
                head_model.func_head.load_state_dict(torch.load(default_best, map_location=device))
                best_path = default_best
                print(f"[INFO] Loaded default best checkpoint: {default_best}")
            elif load_checkpoint:
                print(f"[WARN] Provided checkpoint not found: {load_checkpoint}; proceeding with randomly initialized head.")
        # Ensure mining is enabled if user forgot to set miner_enabled
        if not miner_enabled:
            print("[INFO] mining_only implies miner_enabled=True.")
            miner_enabled = True
    else:
        for epoch in range(num_epochs):
            if shuffle_each_epoch:
                random.shuffle(train_data)

            print(f"\n[INFO] Epoch {epoch+1}/{num_epochs}")
            head_model.train()
            epoch_metrics: List[Dict[str, np.ndarray]] = []

            for idx, ex in tqdm(enumerate(train_data), total=len(train_data)):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = head_model.loss_on_example(
                    ex,
                    only_functoken=only_functoken,
                    inspect=(inspect_token_logits and idx < max(1, int(inspect_limit))),
                    inspect_topk=inspect_topk,
                    inspect_limit=inspect_limit,
                )
                loss.backward()
                optimizer.step()
                global_step += 1

                epoch_metrics.append(metrics)

                # Step logging: loss
                if use_wandb and wandb is not None:
                    wandb.log({"train/loss": float(loss.item()), "step": int(global_step)})

                if (idx + 1) % 20 == 0:
                    agg = aggregate_metrics(epoch_metrics)
                    print(f"[TRN] step={global_step} loss={loss.item():.4f} P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f}")
                    if use_wandb and wandb is not None:
                        wandb.log({"train/precision": agg["precision"], "train/recall": agg["recall"], "train/f1": agg["f1"], "step": int(global_step)})

            # End epoch: evaluate on val
            head_model.eval()
            val_batches: List[Dict[str, np.ndarray]] = []
            with torch.no_grad():
                for ex in tqdm(val_data, total=len(val_data)):
                    loss, metrics = head_model.loss_on_example(ex, only_functoken=only_functoken)
                    val_batches.append(metrics)
            val_agg = aggregate_metrics(val_batches)
            print(f"[VAL] P={val_agg['precision']:.3f} R={val_agg['recall']:.3f} F1={val_agg['f1']:.3f} on {len(val_data)} examples")
            if use_wandb and wandb is not None:
                wandb.log(
                    {
                        "val/precision": val_agg["precision"],
                        "val/recall": val_agg["recall"],
                        "val/f1": val_agg["f1"],
                        "epoch": int(epoch + 1),
                    }
                )

            # Optionally log epoch-level macro (over function classes), computed from aggregated counts
            K = len(func_dict)
            tp_c, pred_c, true_c = sum_class_counters(epoch_metrics, K)
            prec_c, rec_c, f1_c, p_macro, r_macro, f1_macro = per_class_metrics(tp_c, pred_c, true_c)
            tp_v, pred_v, true_v = sum_class_counters(val_batches, K)
            prec_v, rec_v, f1_v, pM_v, rM_v, f1M_v = per_class_metrics(tp_v, pred_v, true_v)
            print(f"[TRN-CLASS] macro P={p_macro:.3f} R={r_macro:.3f} F1={f1_macro:.3f}")
            print(f"[VAL-CLASS] macro P={pM_v:.3f} R={rM_v:.3f} F1={f1M_v:.3f}")
            if use_wandb and wandb is not None:
                wandb.log({"train/macro_precision": p_macro, "train/macro_recall": r_macro, "train/macro_f1": f1_macro, "epoch": int(epoch + 1)})
                wandb.log({"val/macro_precision": pM_v, "val/macro_recall": rM_v, "val/macro_f1": f1M_v, "epoch": int(epoch + 1)})

            # Save head weights every epoch; track best by val F1
            save_path = os.path.join(save_dir, f"head_epoch_{epoch}.pth")
            torch.save(head_model.func_head.state_dict(), save_path)
            print(f"[INFO] Saved head weights to {save_path}")
            if val_agg["f1"] > best_val_f1:
                best_val_f1 = val_agg["f1"]
                best_path = os.path.join(save_dir, "head_best.pth")
                torch.save(head_model.func_head.state_dict(), best_path)
                print(f"[INFO] New best F1={best_val_f1:.3f}; saved to {best_path}")
                if use_wandb and wandb is not None:
                    wandb.log({"val/best_f1": best_val_f1, "best_checkpoint": best_path, "epoch": int(epoch + 1)})

        print("\n[INFO] Training complete.")
        if best_path:
            print(f"[INFO] Best head checkpoint: {best_path} (F1={best_val_f1:.3f})")

    # ===========================
    # Post-training mining phase
    # ===========================
    if miner_enabled:
        if best_path is None:
            print("[POST-MINER] No best checkpoint found; using final epoch head weights.")
        else:
            # Load best checkpoint weights before mining
            head_model.func_head.load_state_dict(torch.load(best_path, map_location=device))
            print(f"[POST-MINER] Loaded best head weights from {best_path}")
        head_model.eval()
        # Overwrite output file (fresh run)
        if os.path.exists(post_mine_output):
            try:
                os.remove(post_mine_output)
            except Exception:
                print(f"[POST-MINER][WARN] Could not remove existing {post_mine_output}; will append.")
    # (OLD behavior removed) Previously we appended raw training examples with a different schema.
    # We now unify schema: when append_train_raw=True we generate mined-style TOOL rows for each
    # supervised function start in the training split (no REJ rows) so downstream reranker code
    # receives consistent fields (candidates, gold_label, etc.).

        mined_rows = 0
        mined_rej = 0
        mined_tool = 0
        with torch.no_grad():
            for ex_idx, ex in tqdm(enumerate(val_data), total=len(val_data), desc="Post-mine val"):
                # --- Forward prep -------------------------------------------------
                # Tokenize + build function start labels (same logic as training)
                text_ids = head_model._encode_with_specials(ex["text"]) if isinstance(ex.get("text"), str) else []
                if not text_ids:
                    continue
                input_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                labels_list, _ = head_model.build_labels(ex, text_ids, debug=False)
                labels = torch.tensor(labels_list, dtype=torch.long, device=device).unsqueeze(0)
                outputs = head_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                token_logits_full = outputs.logits
                func_logits_full = head_model.func_head(hidden.float())
                V = int(token_logits_full.size(-1))
                K_local = len(head_model.func_dict)
                if K_local == 0:
                    break
                token_logits = token_logits_full[:, :-1, :].float()
                func_logits = func_logits_full[:, :-1, :].float()
                next_tokens = input_ids[:, 1:]
                joint_targets = next_tokens.clone()
                # Align tool-start supervision with next token (same off-by-one fix as training)
                func_labels_next = labels[:, 1:]
                func_mask = func_labels_next != -100
                if func_mask.any():
                    joint_targets[func_mask] = V + func_labels_next[func_mask]
                # --- Distributions -------------------------------------------------
                # Slice softmax over functions (still used ONLY for candidate ordering / diagnostics)
                func_probs_all = F.softmax(func_logits[0], dim=-1)  # [T-1, K]
                entropy_all = -torch.sum(func_probs_all * torch.log(func_probs_all + 1e-12), dim=-1)
                Tm1 = func_probs_all.size(0)
                rows: List[str] = []
                for t in range(Tm1):
                    gold_target = int(joint_targets[0, t].item())            # ID in joint space (token or token+K offset)
                    gold_is_tool = gold_target >= V                          # Whether gold next token is a tool start
                    probs_step = func_probs_all[t]                            # Function slice distribution (diagnostics only)
                    k_eff = min(int(post_mine_top_k), K_local)
                    tk_prob, tk_id = torch.topk(probs_step, k=k_eff)         # Top-k candidate tools (slice space)
                    entropy = float(entropy_all[t].item())
                    cand_ids = [int(i.item()) for i in tk_id]
                    cand_names = [head_model.id2func.get(i, str(i)) for i in cand_ids]

                    # Raw logits for gating (joint competition intuition)
                    tool_logits_step = func_logits[0, t]                     # (K,)
                    token_logits_step = token_logits[0, t]                   # (V,)
                    max_tool_logit, top_tool_id = torch.max(tool_logits_step, dim=-1)
                    max_word_logit, _ = torch.max(token_logits_step, dim=-1)
                    joint_tool_beats_token = bool(max_tool_logit.item() > max_word_logit.item())
                    logit_margin = float(max_tool_logit.item() - max_word_logit.item())

                    # Joint rank of the top tool inside concatenated logits (1 = highest overall)
                    # (Efficient enough here given small mining set; avoids sorting all if not needed.)
                    joint_step = torch.cat([token_logits_step, tool_logits_step], dim=0)
                    # Count how many logits strictly greater than the top tool logit
                    higher = torch.sum(joint_step > max_tool_logit).item()
                    joint_rank_tool_top1 = int(higher + 1)

                    gold_label = None
                    record = False
                    if not gold_is_tool:
                        # REJ condition: tool truly outranks every token in raw logit space
                        if joint_tool_beats_token:
                            gold_label = "REJ"
                            record = True
                    else:
                        # Tool disambiguation: always record, force include gold tool if absent
                        gold_tool_id = int(gold_target - V)
                        gold_tool_name = head_model.id2func.get(gold_tool_id, str(gold_tool_id))
                        if gold_tool_id not in cand_ids:
                            cand_ids.append(gold_tool_id)
                            cand_names.append(gold_tool_name)
                        gold_label = gold_tool_name
                        record = True

                    if not record:
                        continue  # skip (no REJ & no gold tool)

                    prefix_tokens = input_ids[0, : t + 1].tolist()
                    try:
                        prefix_text = head_model.tokenizer.decode(prefix_tokens, skip_special_tokens=False)
                    except Exception:
                        prefix_text = ""

                    row = {
                        "split": "val",
                        "example_index": int(ex_idx),
                        "position": t,
                        "prefix_text": prefix_text,
                        "candidates": cand_names,
                        "candidate_tool_ids": cand_ids,
                        "candidate_probs": [float(probs_step[i].item()) for i in cand_ids],  # slice softmax (diagnostic)
                        "gold_label": gold_label,
                        "gold_is_tool": bool(gold_is_tool),
                        # New diagnostic fields (strict gating basis)
                        "max_tool_logit": float(max_tool_logit.item()),
                        "max_word_logit": float(max_word_logit.item()),
                        "logit_margin": logit_margin,
                        "joint_rank_tool_top1": joint_rank_tool_top1,
                        "joint_tool_beats_token": joint_tool_beats_token,
                        # Retained diagnostics
                        "entropy": entropy,
                        "k": int(post_mine_top_k),
                    }
                    rows.append(json.dumps(row, ensure_ascii=False))
                    mined_rows += 1
                    if gold_label == "REJ":
                        mined_rej += 1
                    else:
                        mined_tool += 1
                if rows:
                    with open(post_mine_output, "a", encoding="utf-8") as f_out:
                        f_out.write("\n".join(rows) + "\n")
        if mined_rows > 0:
            print(f"[POST-MINER] Completed (val): rows={mined_rows} tool_rows={mined_tool} rej_rows={mined_rej} avg_rows_per_example={mined_rows/len(val_data):.2f}")
        else:
            print("[POST-MINER] No rows mined on validation (check thresholds or data).")

        # Append unified-schema TRAIN tool rows (no REJ) if requested
        if append_train_raw:
            train_tool_rows = 0
            train_examples_used = 0
            with torch.no_grad():
                for tr_idx, ex in tqdm(enumerate(train_data), total=len(train_data), desc="Append train tool rows"):
                    text = ex.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    text_ids = head_model._encode_with_specials(text)
                    if not text_ids:
                        continue
                    input_ids = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
                    attention_mask = torch.ones_like(input_ids)
                    labels_list, _ = head_model.build_labels(ex, text_ids, debug=False)
                    labels = torch.tensor(labels_list, dtype=torch.long, device=device).unsqueeze(0)
                    outputs = head_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    token_logits_full = outputs.logits
                    func_logits_full = head_model.func_head(hidden.float())
                    Vt = int(token_logits_full.size(-1))
                    token_logits = token_logits_full[:, :-1, :].float()
                    func_logits = func_logits_full[:, :-1, :].float()
                    func_probs_all = F.softmax(func_logits[0], dim=-1)
                    next_tokens = input_ids[:, 1:]
                    func_labels_next = labels[:, 1:]
                    func_mask = func_labels_next != -100
                    if not func_mask.any():
                        continue  # no supervised tool positions
                    positions = torch.nonzero(func_mask[0], as_tuple=False).view(-1).tolist()
                    rows_out: List[str] = []
                    for p in positions:
                        gold_tool_id = int(func_labels_next[0, p].item())
                        gold_tool_name = head_model.id2func.get(gold_tool_id, str(gold_tool_id))
                        probs_step = func_probs_all[p]
                        k_eff = min(int(post_mine_top_k), len(head_model.func_dict))
                        tk_prob, tk_id = torch.topk(probs_step, k=k_eff)
                        cand_ids = [int(i.item()) for i in tk_id]
                        cand_names = [head_model.id2func.get(i, str(i)) for i in cand_ids]
                        if gold_tool_id not in cand_ids:
                            cand_ids.append(gold_tool_id)
                            cand_names.append(gold_tool_name)
                        tool_logits_step = func_logits[0, p]
                        token_logits_step = token_logits[0, p]
                        max_tool_logit, _ = torch.max(tool_logits_step, dim=-1)
                        max_word_logit, _ = torch.max(token_logits_step, dim=-1)
                        logit_margin = float(max_tool_logit.item() - max_word_logit.item())
                        prefix_tokens = input_ids[0, : p + 1].tolist()
                        try:
                            prefix_text = head_model.tokenizer.decode(prefix_tokens, skip_special_tokens=False)
                        except Exception:
                            prefix_text = ""
                        row = {
                            "split": "train",
                            "example_index": int(tr_idx),
                            "position": int(p),
                            "prefix_text": prefix_text,
                            "candidates": cand_names,
                            "candidate_tool_ids": cand_ids,
                            "candidate_probs": [float(probs_step[i].item()) for i in cand_ids],
                            "gold_label": gold_tool_name,
                            "gold_is_tool": True,
                            "max_tool_logit": float(max_tool_logit.item()),
                            "max_word_logit": float(max_word_logit.item()),
                            "logit_margin": logit_margin,
                            "joint_rank_tool_top1": None,  # optional (skipped to save time)
                            "joint_tool_beats_token": bool(max_tool_logit.item() > max_word_logit.item()),
                            "entropy": float((-torch.sum(probs_step * torch.log(probs_step + 1e-12))).item()),
                            "k": int(post_mine_top_k),
                            "is_augmented_train": True,
                        }
                        rows_out.append(json.dumps(row, ensure_ascii=False))
                        train_tool_rows += 1
                    if rows_out:
                        with open(post_mine_output, "a", encoding="utf-8") as f_out:
                            f_out.write("\n".join(rows_out) + "\n")
                        train_examples_used += 1
            print(f"[POST-MINER] Appended train tool rows: {train_tool_rows} from {train_examples_used} train examples (append_train_raw=True).")



if __name__ == "__main__":
    fire.Fire(main)