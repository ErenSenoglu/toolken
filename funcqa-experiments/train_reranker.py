import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer  # optional separate tokenizer path

# Reuse seeding & model loader logic by lightly importing from train_plus (keeps consistency)
# Robust import of utilities when running as a standalone script (no package context)
try:  # package-style
    from .train_plus import set_seeds, load_gemma_model  # type: ignore
except Exception:  # direct sibling import fallback
    try:
        from train_plus import set_seeds, load_gemma_model  # type: ignore
    except Exception:
        import sys as _sys, os as _os
        _sys.path.append(_os.path.dirname(__file__))
        from train_plus import set_seeds, load_gemma_model  # type: ignore


# ------------------------------------------------------------------------------------
# Reranker Head: lightweight embedding projection over candidate class subset + REJ
# ------------------------------------------------------------------------------------
class RerankerHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.W = nn.Embedding(num_classes, hidden_size)
        nn.init.normal_(self.W.weight, mean=0.0, std=0.02)

    def forward_logits(self, h: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        """
        h: (B, H)      last hidden state vector per example
        cand_ids: (B, Kc) class ids in global (tool+REJ) space
        returns: (B, Kc) logits
        """
        W_cand = self.W(cand_ids)          # (B, Kc, H)
        # batch dot: (B, H) x (B, Kc, H) -> (B, Kc)
        logits = torch.einsum("bd,bkd->bk", h, W_cand)
        return logits

    def forward_logits_masked(self, h: torch.Tensor, cand_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked variant: avoids embedding padded slots (cand_ids == -1).

        h: (B, H)
        cand_ids: (B, Kc) with -1 for padded positions
        mask: (B, Kc) bool where True means valid candidate
        returns: (B, Kc) logits with -1e30 in padded positions (never contribute to loss)
        """
        B, Kc = cand_ids.shape
        device = cand_ids.device
        dtype = h.dtype  # preserve (e.g., bfloat16) for speed
        logits_full = torch.full((B, Kc), -1e30, device=device, dtype=dtype)
        if mask.any():
            flat_idx = torch.nonzero(mask, as_tuple=False)
            b_idx = flat_idx[:, 0]
            k_idx = flat_idx[:, 1]
            valid_ids = cand_ids[b_idx, k_idx]
            emb = self.W(valid_ids)
            if emb.dtype != dtype:
                emb = emb.to(dtype)
            h_sel = h[b_idx]
            if h_sel.dtype != emb.dtype:
                h_sel = h_sel.to(emb.dtype)
            dot = (h_sel * emb).sum(dim=-1)
            logits_full[b_idx, k_idx] = dot
        return logits_full


# ------------------------------------------------------------------------------------
# Prompt Builder
# ------------------------------------------------------------------------------------
def build_prompt(prefix_text: str,
                 candidates: List[str],
                 tool_docs: Dict[str, str],
                 style: str = "plain",
                 rej_label: str = "REJ",
                 rej_doc: str = "continue the answer without calling any tool.") -> str:
    """Construct a reranking prompt incorporating tool descriptions.

    style == "plain": simple system/instruction layout.
    style == "llama2_chat": approximate Llama2 chat format with [INST] tags.
    """
    lines = []
    if style == "plain":
        lines.append("System:\nYou are a helpful AI assistant. Your task is to analyze the context and select the most appropriate tool or action from the options below. Choose exactly one option that best addresses the user's needs.\n")
        lines.append("Tools (in descending relevance from the base model):")
        for i, t in enumerate(candidates, start=1):
            doc = tool_docs.get(t, f"{t}(...) usage.")
            lines.append(f"{i}) {t}: {doc}")
        lines.append(f"{len(candidates)+1}) {rej_label}: {rej_doc}")
        # Reply instruction
        opts = " or ".join(candidates + [rej_label])
        lines.append("\nInstruction:\nGiven the context, choose EXACTLY ONE option from the list above.\n"
                     f"Reply with ONLY the option token EXACTLY as written: {opts}\n")
        lines.append("Context:")
        lines.append(prefix_text)
        lines.append("\nAnswer:\n")
        return "\n".join(lines)
    else:  # llama2_chat style (simplified)
        # We'll craft a single user prompt inside [INST] ... [/INST]
        tool_lines = []
        for i, t in enumerate(candidates, start=1):
            doc = tool_docs.get(t, f"{t}(...) usage.")
            tool_lines.append(f"{i}) {t}: {doc}")
        tool_lines.append(f"{len(candidates)+1}) {rej_label}: {rej_doc}")
        opts = " or ".join(candidates + [rej_label])
        user = ("You are a tool selector. Choose the next action.\n" +
                "Tools (in descending relevance from the base model):\n" +
                "\n".join(tool_lines) +
                "\n\nInstruction: Choose EXACTLY ONE option from above. "
                f"Reply with ONLY: {opts}\n" +
                "Context:\n" + prefix_text + "\nAnswer:")
        return f"<s>[INST] {user} [/INST]"


# ------------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------------
@dataclass
class MinedRow:
    prefix_text: str
    candidates: List[str]
    gold_label: str


class MinedJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[MinedRow] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Required fields from schema
                pref = obj.get("prefix_text", "")
                cands = obj.get("candidates", []) or []
                gold = obj.get("gold_label", "REJ")
                # Deduplicate candidates preserving order (force-inclusion earlier already handled upstream)
                seen = set()
                dedup = []
                for c in cands:
                    if c not in seen:
                        seen.add(c)
                        dedup.append(c)
                self.rows.append(MinedRow(pref, dedup, gold))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# ------------------------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------------------------
def make_collate_fn(tokenizer,
                    toolname2idx: Dict[str, int],
                    tool_docs: Dict[str, str],
                    style: str,
                    max_length: int,
                    tool_cap: int,
                    rej_label: str = "REJ",
                    rej_doc: str = "continue the answer without calling any tool."):
    def collate(batch: List[MinedRow]):
        prompts: List[str] = []
        cand_id_lists: List[List[int]] = []
        targets: List[int] = []
        for row in batch:
            # Build candidate list; FORCE-INCLUDE gold tool if it's a tool and missing.
            candidates = list(row.candidates)
            # Keep only top 'tool_cap' tool candidates (exclude REJ and extras)
            candidates = [c for c in candidates if c != rej_label][:tool_cap]
            # Optional policy: If gold tool outside truncated set we do NOT force-add (mirrors inference constraint).
            if row.gold_label != rej_label and row.gold_label in toolname2idx and row.gold_label in row.candidates and row.gold_label not in candidates:
                # This branch will be rare (gold outside truncated top-2) -> we skip adding to preserve strict cap.
                pass
            # Append REJ sentinel if absent (never part of tool docs enumeration when building prompt candidates)
            if rej_label not in candidates:
                candidates_plus = candidates + [rej_label]
            else:
                candidates_plus = candidates
            # Prompt should list only tool candidates (exclude REJ which is appended separately in numbering logic inside builder)
            prompt = build_prompt(row.prefix_text, [c for c in candidates if c != rej_label], tool_docs, style=style, rej_label=rej_label, rej_doc=rej_doc)
            prompts.append(prompt)
            # Map candidate names to global class ids
            cand_ids = [toolname2idx[c] for c in candidates_plus]
            cand_id_lists.append(cand_ids)
            # Target index inside this candidate list
            if row.gold_label in candidates_plus:
                gold_local = candidates_plus.index(row.gold_label)
            else:
                # Fallback: label not recognized -> map to REJ (should be rare after force-inclusion step)
                gold_local = candidates_plus.index(rej_label)
            targets.append(gold_local)

        # Tokenize all prompts
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        # Build candidate id tensor (ragged -> pad with -1 then mask) but we only need per-row set; use list of tensors
        max_c = max(len(l) for l in cand_id_lists)
        cand_tensor = torch.full((len(batch), max_c), fill_value=-1, dtype=torch.long)
        mask = torch.zeros((len(batch), max_c), dtype=torch.bool)
        for i, l in enumerate(cand_id_lists):
            cand_tensor[i, :len(l)] = torch.tensor(l, dtype=torch.long)
            mask[i, :len(l)] = True
        target_tensor = torch.tensor(targets, dtype=torch.long)
        return enc, cand_tensor, mask, target_tensor
    return collate


# ------------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------------
def train_reranker(
    mined_jsonl: str = "miner_half_split.jsonl",
    model_name_or_path: str = "/workspace/.hf_home/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    tokenizer_name_or_path: Optional[str] = None,
    func_dict: str = "./data/funcqa/func_dict.json",
    tool_docs_json: Optional[str] = "./data/funcqa/tool_docs.json",
    style: str = "plain",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "reranker_head_best.pt",
    max_length: int = 1024,
    val_fraction: float = 0.1,
    seed: int = 42,
    class_balance: str = "none",   # options: none, reweight
    resample_strategy: str = "none",  # options: none, oversample_tool
    target_tool_fraction: float = 0.5,  # desired fraction of tool rows after resampling (only for oversample_tool)
    log_every: int = 10,
    retrieve_all_layers: bool = False,  # if True, keep output_hidden_states (higher memory)
    empty_cache_each_epoch: bool = True,
    eval_fix_rates: bool = True,
    fix_rates_output: str = "reranker_fix_eval_best.json",
    train_top_k_tools: int = 3,  # number of tool candidates (excluding REJ)
):
    """Train a lightweight reranker head over mined tool selection rows.

    class_balance:
        none     - standard uniform batching
        reweight - apply inverse frequency weights to CE loss
    """
    assert os.path.exists(mined_jsonl), f"mined_jsonl not found: {mined_jsonl}"
    assert os.path.exists(func_dict), f"func_dict not found: {func_dict}"
    if tool_docs_json:
        assert os.path.exists(tool_docs_json), f"tool_docs_json not found: {tool_docs_json}"

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tool mapping
    with open(func_dict, "r", encoding="utf-8") as f:
        base_func_dict = json.load(f)
    toolname2idx = dict(base_func_dict)
    if "REJ" not in toolname2idx:
        toolname2idx["REJ"] = max(toolname2idx.values(), default=-1) + 1
    num_tool_classes = len(toolname2idx)

    # Optional: persist the extended vocab map alongside head weights later
    tools_vocab_path = os.path.join(os.path.dirname(save_path) or ".", "tools_vocab_reranker.json")

    # Load tool docs
    tool_docs = {}
    if tool_docs_json:
        with open(tool_docs_json, "r", encoding="utf-8") as f:
            tool_docs = json.load(f)
    # Ensure every tool has at least a stub doc
    for name in toolname2idx.keys():
        if name == "REJ":
            continue
        tool_docs.setdefault(name, f"{name}(...) invocation." )

    # Dataset + deterministic shuffled split (avoid order bias from grouped JSONL)
    dataset_all = MinedJsonlDataset(mined_jsonl)
    n_total = len(dataset_all)
    n_val = max(1, int(n_total * val_fraction)) if n_total > 1 else 0
    n_train = n_total - n_val
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_idx_set = set(indices[:n_val])
    train_subset = [dataset_all.rows[i] for i in indices[n_val:]]
    val_subset = [dataset_all.rows[i] for i in indices[:n_val]] if n_val > 0 else []

    # Count how many rows lose their gold tool under top-k truncation (gold != REJ and not in first k tools)
    def count_skipped(rows: List[MinedRow]) -> int:
        skipped = 0
        for r in rows:
            if r.gold_label != "REJ":
                topk = [c for c in r.candidates if c != "REJ"][:train_top_k_tools]
                if r.gold_label not in topk:
                    skipped += 1
        return skipped
    skipped_gold_train = count_skipped(train_subset)
    skipped_gold_val = count_skipped(val_subset) if val_subset else 0

    # Optional resampling to combat collapse toward REJ
    if resample_strategy == "oversample_tool":
        tool_rows = [r for r in train_subset if r.gold_label != "REJ"]
        rej_rows = [r for r in train_subset if r.gold_label == "REJ"]
        n_tool = len(tool_rows)
        n_rej = len(rej_rows)
        if n_tool > 0 and n_rej > 0:
            current_fraction = n_tool / max(n_tool + n_rej, 1)
            desired_total = n_tool + n_rej
            # Compute target total tool count given current total (adjust iteratively later)
            target_tool_count = int(target_tool_fraction * (n_tool + n_rej))
            if target_tool_count <= n_tool:
                # Already at or above desired fraction (cannot downsample in oversample mode)
                pass
            else:
                needed = target_tool_count - n_tool
                rng2 = random.Random(seed + 123)
                extra = [tool_rows[i % n_tool] for i in range(needed)]
                rng2.shuffle(extra)
                tool_rows_extended = tool_rows + extra
                train_subset = rej_rows + tool_rows_extended
                rng2.shuffle(train_subset)
                print(f"[RESAMPLE] Oversampled tool rows: tool {n_tool}->{len(tool_rows_extended)} | rej {n_rej} | target_tool_fraction={target_tool_fraction}")
        else:
            print("[RESAMPLE][WARN] Skipping oversample_tool (one of the classes is empty).")

    class DistWrap(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return self.rows[idx]

    train_ds = DistWrap(train_subset)
    val_ds = DistWrap(val_subset)

    # Tokenizer & model
    # Load model weights strictly from model_name_or_path.
    base_model, tokenizer = load_gemma_model(model_name_or_path, device=device)
    if tokenizer_name_or_path and tokenizer_name_or_path != model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    # Disable KV caching to lower memory (we only do single forward for CLS-style feature)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    # Freeze
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()

    # Infer hidden size
    cfg = base_model.config
    hidden_size = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        or getattr(getattr(cfg, "text_config", None), "n_embd", None)
    )
    assert hidden_size is not None, "Could not infer hidden size"

    head = RerankerHead(int(hidden_size), num_tool_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    # Compute class frequencies (for optional reweight)
    if class_balance == "reweight":
        freq = torch.zeros(num_tool_classes, dtype=torch.long)
        for r in train_subset:
            gold = r.gold_label if r.gold_label in toolname2idx else "REJ"
            freq[toolname2idx[gold]] += 1
        total = freq.sum().item() if freq.sum().item() > 0 else 1
        inv_freq = total / torch.clamp(freq.float(), min=1.0)
        weights = inv_freq / inv_freq.mean()
        ce_weight = weights.to(device)
        print("[INFO] Class reweighting active. Example weights (global id -> weight):")
        for gid, w in list(enumerate(weights.tolist()))[:10]:
            # Attempt to recover tool name (reverse lookup)
            try:
                name = next(k for k,v in toolname2idx.items() if v == gid)
            except StopIteration:
                name = f"class_{gid}"
            print(f"   {gid} ({name}): {w:.3f}")
    else:
        ce_weight = None

    collate_fn = make_collate_fn(tokenizer, toolname2idx, tool_docs, style, max_length, train_top_k_tools)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if n_val > 0 else None

    print(f"[INFO] Reranker training rows total={n_total} train={len(train_subset)} val={len(val_subset)} | classes={num_tool_classes} | seed={seed}")
    print(f"[INFO] Skipped gold tools due to top-{train_top_k_tools} cap: train={skipped_gold_train} val={skipped_gold_val}")

    global_step = 0
    best_val_f1 = -1.0  # track best micro F1 on validation
    best_ckpt_path = None
    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0
        # Per-class counters for training F1 (global class id space)
        train_tp = torch.zeros(num_tool_classes, dtype=torch.long)
        train_pred_ct = torch.zeros(num_tool_classes, dtype=torch.long)
        train_true_ct = torch.zeros(num_tool_classes, dtype=torch.long)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for batch in pbar:
            enc, cand_tensor, mask, targets = batch
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            cand_tensor = cand_tensor.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                if retrieve_all_layers:
                    # Full causal LM forward returning all hidden states (memory heavier)
                    out = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    h_last = out.hidden_states[-1][:, -1, :]
                else:
                    # Lightweight: call underlying base transformer to get only last hidden state
                    # Many HF CausalLM classes expose the base model at .model or .transformer
                    backbone = getattr(base_model, 'model', None) or getattr(base_model, 'transformer', None)
                    if backbone is not None:
                        core_out = backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_hidden_states=False,
                        )
                        # core_out may be tuple or ModelOutput
                        if hasattr(core_out, 'last_hidden_state'):
                            h_last = core_out.last_hidden_state[:, -1, :]
                        else:
                            h_last = core_out[0][:, -1, :]
                    else:
                        # Fallback: run full model with hidden states then discard others
                        out = base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        h_last = out.hidden_states[-1][:, -1, :]

            # We must ignore padded candidate slots (-1). Build per-row compact scoring by masking.
            # For efficiency we still score all then mask logits to large negative.
            logits_full = head.forward_logits_masked(h_last, cand_tensor, mask)
            # Loss: if class_balance=='reweight', apply per-sample weighting using the
            # target's global class weight (mapped from cand_tensor). Standard CE otherwise.
            if class_balance == "reweight" and ce_weight is not None:
                # Gather global target class ids
                bsz = targets.size(0)
                row_ar = torch.arange(bsz, device=targets.device)
                target_global_ids = cand_tensor[row_ar, targets]
                sample_weights = ce_weight[target_global_ids]  # (B,)
                # Log-softmax over full (masked) logits
                log_probs = F.log_softmax(logits_full, dim=-1)
                picked = log_probs[row_ar, targets]  # (B,)
                loss_vec = -sample_weights * picked
                # Normalize by sum of weights (stable for varying batch compositions)
                loss = loss_vec.sum() / sample_weights.sum().clamp(min=1e-9)
            else:
                loss = F.cross_entropy(logits_full, targets, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping (max norm 1.0) to stabilize training and reduce harmful large updates
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits_full, dim=-1)
                acc = (pred == targets).float().mean().item()
                # Map local candidate index to global class id
                bsz = input_ids.size(0)
                row_ar = torch.arange(bsz, device=pred.device)
                # Gather global ids (masked positions have -1 entries; targets point only to valid slots)
                global_true = cand_tensor[row_ar, targets.cpu()].cpu()
                global_pred = cand_tensor[row_ar, pred.cpu()].cpu()
                for gt, gp in zip(global_true.tolist(), global_pred.tolist()):
                    if gt >= 0:
                        train_true_ct[gt] += 1
                    if gp >= 0:
                        train_pred_ct[gp] += 1
                    if gt == gp and gt >= 0:
                        train_tp[gt] += 1
            bsz = input_ids.size(0)
            running_loss += loss.item() * bsz
            running_acc += acc * bsz
            seen += bsz
            global_step += 1
            if global_step % log_every == 0:
                pbar.set_postfix({"loss": running_loss / seen, "acc": running_acc / seen})

        epoch_loss = running_loss / max(seen, 1)
        epoch_acc = running_acc / max(seen, 1)
        # Compute F1 (macro & micro) for training
        tp_f = train_tp.float()
        pred_f = train_pred_ct.float().clamp(min=1)
        true_f = train_true_ct.float().clamp(min=1)
        f1_per_class = (2 * tp_f) / (pred_f + true_f)
        macro_f1_tr = f1_per_class.mean().item()
        micro_tp = train_tp.sum().item()
        micro_pred = train_pred_ct.sum().item()
        micro_true = train_true_ct.sum().item()
        micro_f1_tr = (2 * micro_tp) / max(micro_pred + micro_true, 1)
        print(f"[EPOCH {epoch}] train_loss={epoch_loss:.4f} acc={epoch_acc:.4f} microF1={micro_f1_tr:.4f} macroF1={macro_f1_tr:.4f}")
        # Collapse warning: if no tool predictions or no tool truths encountered
        if train_pred_ct.sum().item() > 0:
            tool_class_ids = [gid for name, gid in toolname2idx.items() if name != "REJ"]
            tool_pred_total = train_pred_ct[tool_class_ids].sum().item() if tool_class_ids else 0
            tool_true_total = train_true_ct[tool_class_ids].sum().item() if tool_class_ids else 0
            tool_tp_total = train_tp[tool_class_ids].sum().item() if tool_class_ids else 0
            if tool_true_total > 0 and tool_tp_total == 0:
                print("[WARN] Tool recall is 0 this epoch (model predicting only REJ for tool rows). Consider increasing oversampling or other interventions.")

        # Validation
        if val_loader is not None and len(val_loader) > 0:
            head.eval()
            v_loss = 0.0
            v_acc = 0.0
            v_seen = 0
            with torch.no_grad():
                for batch in val_loader:
                    enc, cand_tensor, mask, targets = batch
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    cand_tensor = cand_tensor.to(device)
                    mask = mask.to(device)
                    targets = targets.to(device)
                    if retrieve_all_layers:
                        out = base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        h_last = out.hidden_states[-1][:, -1, :]
                    else:
                        backbone = getattr(base_model, 'model', None) or getattr(base_model, 'transformer', None)
                        if backbone is not None:
                            core_out = backbone(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                output_hidden_states=False,
                            )
                            if hasattr(core_out, 'last_hidden_state'):
                                h_last = core_out.last_hidden_state[:, -1, :]
                            else:
                                h_last = core_out[0][:, -1, :]
                        else:
                            out = base_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True,
                                use_cache=False,
                            )
                            h_last = out.hidden_states[-1][:, -1, :]
                    logits_full = head.forward_logits_masked(h_last, cand_tensor, mask)
                    loss = F.cross_entropy(logits_full, targets, reduction="mean")
                    pred = torch.argmax(logits_full, dim=-1)
                    acc = (pred == targets).float().mean().item()
                    bsz = input_ids.size(0)
                    v_loss += loss.item() * bsz
                    v_acc += acc * bsz
                    v_seen += bsz
                    # Accumulate per-class counts (global ids)
                    row_ar = torch.arange(bsz, device=pred.device)
                    global_true = cand_tensor[row_ar, targets.cpu()].cpu()
                    global_pred = cand_tensor[row_ar, pred.cpu()].cpu()
                    # Initialize outside loop if first batch
                    if 'val_tp' not in locals():
                        val_tp = torch.zeros(num_tool_classes, dtype=torch.long)
                        val_pred_ct = torch.zeros(num_tool_classes, dtype=torch.long)
                        val_true_ct = torch.zeros(num_tool_classes, dtype=torch.long)
                    for gt, gp in zip(global_true.tolist(), global_pred.tolist()):
                        if gt >= 0:
                            val_true_ct[gt] += 1
                        if gp >= 0:
                            val_pred_ct[gp] += 1
                        if gt == gp and gt >= 0:
                            val_tp[gt] += 1
            v_loss_mean = v_loss / max(v_seen, 1)
            v_acc_mean = v_acc / max(v_seen, 1)
            if 'val_tp' in locals():
                v_tp_f = val_tp.float()
                v_pred_f = val_pred_ct.float().clamp(min=1)
                v_true_f = val_true_ct.float().clamp(min=1)
                v_f1_per_class = (2 * v_tp_f) / (v_pred_f + v_true_f)
                macro_f1_val = v_f1_per_class.mean().item()
                micro_tp_v = val_tp.sum().item()
                micro_pred_v = val_pred_ct.sum().item()
                micro_true_v = val_true_ct.sum().item()
                micro_f1_val = (2 * micro_tp_v) / max(micro_pred_v + micro_true_v, 1)
                print(f"[EPOCH {epoch}] val_loss={v_loss_mean:.4f} val_acc={v_acc_mean:.4f} microF1={micro_f1_val:.4f} macroF1={macro_f1_val:.4f}")
                # Save best (micro F1) checkpoint
                if micro_f1_val > best_val_f1:
                    best_val_f1 = micro_f1_val
                    root_dir = os.path.dirname(save_path) or "."
                    os.makedirs(root_dir, exist_ok=True)
                    best_ckpt_path = os.path.join(root_dir, os.path.splitext(os.path.basename(save_path))[0] + "_best.pt")
                    torch.save({
                        "state_dict": head.state_dict(),
                        "toolname2idx": toolname2idx,
                        "hidden_size": hidden_size,
                        "style": style,
                        "model_name_or_path": model_name_or_path,
                        "best_val_micro_f1": best_val_f1,
                        "epoch": epoch,
                        "train_top_k_tools": train_top_k_tools,
                    }, best_ckpt_path)
                    print(f"[BEST] New best val microF1={best_val_f1:.4f} saved to {best_ckpt_path}")
            else:
                print(f"[EPOCH {epoch}] val_loss={v_loss_mean:.4f} val_acc={v_acc_mean:.4f} microF1=0.0000 macroF1=0.0000")

        if empty_cache_each_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save artifacts after training loop
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "toolname2idx": toolname2idx,
        "hidden_size": hidden_size,
        "style": style,
        "model_name_or_path": model_name_or_path,
    "train_top_k_tools": train_top_k_tools,
    }, save_path)
    with open(tools_vocab_path, "w", encoding="utf-8") as f:
        json.dump(toolname2idx, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Reranker head saved to {save_path}\n[SAVE] Extended vocab saved to {tools_vocab_path}")
    if best_ckpt_path:
        print(f"[SAVE] Best validation microF1 checkpoint at {best_ckpt_path} (microF1={best_val_f1:.4f})")

    # ---------------------------------------------
    # Post-training fix-rate evaluation (optional)
    # ---------------------------------------------
    if eval_fix_rates:
        print("[FIX-EVAL] Starting fix-rate evaluation on mined rows...")
        # Decide which head weights to use: best (if exists) else current
        eval_state_dict = None
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            try:
                ckpt_obj = torch.load(best_ckpt_path, map_location="cpu")
                eval_state_dict = ckpt_obj.get("state_dict", ckpt_obj)
                head.load_state_dict(eval_state_dict)
                print(f"[FIX-EVAL] Loaded best checkpoint: {best_ckpt_path}")
            except Exception as e:
                print(f"[FIX-EVAL][WARN] Failed loading best checkpoint ({e}); using final head.")
        head.eval()
        # Counters
        total_rows = 0
        total_rej = 0
        total_tool = 0
        baseline_tool_correct = 0
        reranker_rej_correct = 0
        reranker_tool_correct = 0
        fix_tool_count = 0  # baseline wrong -> reranker correct (tool rows)
        skipped = 0
        # Simple batching over raw rows
        BATCH = max(1, batch_size)
        rows_all = dataset_all.rows
        for start in range(0, len(rows_all), BATCH):
            batch_rows = rows_all[start:start+BATCH]
            # Build prompts & candidate mappings similar to collate (but keep original first candidate for baseline)
            prompts = []
            batch_candidates_plus = []  # list of (candidates_plus, gold_label)
            for r in batch_rows:
                orig_cands = list(r.candidates)
                gold = r.gold_label
                # Baseline correctness for tool rows computed later (need orig_cands[0])
                # Build train-style candidate list
                cands = [c for c in orig_cands if c != "REJ"][:train_top_k_tools]  # enforce top-k tools
                # DO NOT force-add gold if outside top-2 to stay consistent with inference constraint.
                # Append REJ sentinel for scoring
                if "REJ" not in cands:
                    cands_plus = cands + ["REJ"]
                else:
                    cands_plus = cands
                prompt = build_prompt(r.prefix_text, [c for c in cands if c != "REJ"], tool_docs, style=style, rej_label="REJ")
                prompts.append(prompt)
                batch_candidates_plus.append((cands_plus, gold, orig_cands))
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                if retrieve_all_layers:
                    out = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    h_last = out.hidden_states[-1][:, -1, :]
                else:
                    backbone = getattr(base_model, 'model', None) or getattr(base_model, 'transformer', None)
                    if backbone is not None:
                        core_out = backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_hidden_states=False,
                        )
                        if hasattr(core_out, 'last_hidden_state'):
                            h_last = core_out.last_hidden_state[:, -1, :]
                        else:
                            h_last = core_out[0][:, -1, :]
                    else:
                        out = base_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        h_last = out.hidden_states[-1][:, -1, :]
            # Score each row separately (variable candidate counts)
            for bi, (cands_plus, gold, orig_cands) in enumerate(batch_candidates_plus):
                if not cands_plus:
                    skipped += 1
                    continue
                total_rows += 1
                gold_is_rej = (gold == "REJ")
                if gold_is_rej:
                    total_rej += 1
                else:
                    total_tool += 1
                # Baseline correctness
                if not gold_is_rej:
                    if orig_cands and orig_cands[0] == gold:
                        baseline_tool_correct += 1
                # Build logits for this example
                cand_ids = torch.tensor([toolname2idx[c] for c in cands_plus], device=device).unsqueeze(0)  # (1,Kc)
                mask = torch.ones_like(cand_ids, dtype=torch.bool)
                logits = head.forward_logits_masked(h_last[bi:bi+1], cand_ids, mask)  # (1,Kc)
                pred_idx = int(torch.argmax(logits[0]).item())
                pred_label = cands_plus[pred_idx]
                # Reranker correctness and fix logic
                if gold_is_rej:
                    if pred_label == "REJ":
                        reranker_rej_correct += 1
                else:
                    if pred_label == gold:
                        reranker_tool_correct += 1
                        # fix only if baseline was wrong
                        baseline_was_wrong = not (orig_cands and orig_cands[0] == gold)
                        if baseline_was_wrong:
                            fix_tool_count += 1
        # Final metrics
        baseline_total_correct = baseline_tool_correct  # REJ rows baseline always wrong
        reranker_total_correct = reranker_rej_correct + reranker_tool_correct
        tool_baseline_wrong = total_tool - baseline_tool_correct
        fix_rej_rate = (reranker_rej_correct / total_rej) if total_rej > 0 else 0.0
        fix_tool_rate = (fix_tool_count / tool_baseline_wrong) if tool_baseline_wrong > 0 else 0.0
        baseline_tool_acc = (baseline_tool_correct / total_tool) if total_tool > 0 else 0.0
        reranker_tool_acc = (reranker_tool_correct / total_tool) if total_tool > 0 else 0.0
        reranker_rej_acc = (reranker_rej_correct / total_rej) if total_rej > 0 else 0.0
        baseline_total_acc = baseline_total_correct / max(total_rows, 1)
        reranker_total_acc = reranker_total_correct / max(total_rows, 1)
        net_delta_acc = reranker_total_acc - baseline_total_acc
        summary = {
            "total_rows": total_rows,
            "total_REJ_rows": total_rej,
            "total_TOOL_rows": total_tool,
            "baseline_tool_accuracy": round(baseline_tool_acc, 6),
            "baseline_total_accuracy": round(baseline_total_acc, 6),
            "reranker_tool_accuracy": round(reranker_tool_acc, 6),
            "reranker_REJ_accuracy": round(reranker_rej_acc, 6),
            "reranker_total_accuracy": round(reranker_total_acc, 6),
            "delta_tool_accuracy": round(reranker_tool_acc - baseline_tool_acc, 6),
            "fix_REJ_rate": round(fix_rej_rate, 6),
            "fix_TOOL_rate": round(fix_tool_rate, 6),
            "fix_TOOL_count": int(fix_tool_count),
            "reranker_REJ_correct": int(reranker_rej_correct),
            "reranker_TOOL_correct": int(reranker_tool_correct),
            "baseline_TOOL_correct": int(baseline_tool_correct),
            "tool_baseline_wrong": int(tool_baseline_wrong),
            "net_delta_accuracy": round(net_delta_acc, 6),
            "skipped_rows": int(skipped),
            "skipped_gold_train": int(skipped_gold_train),
            "skipped_gold_val": int(skipped_gold_val),
            "used_checkpoint": best_ckpt_path if best_ckpt_path else "final",
            "train_top_k_tools": train_top_k_tools,
        }
        print("[FIX-EVAL] Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        try:
            with open(fix_rates_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[FIX-EVAL] Wrote summary to {fix_rates_output}")
        except Exception as e:
            print(f"[FIX-EVAL][WARN] Could not write summary file: {e}")


def main(**kwargs):  # fire entrypoint
    train_reranker(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
