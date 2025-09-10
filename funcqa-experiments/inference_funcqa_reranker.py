"""Toolken+ (Algorithm 1) inference for FuncQA.

Implements the control flow described by the user:

At each decoding step i:
 1. Run the base LM to obtain the next-token logits over words V.
 2. Run the lightweight function/tool head to obtain logits over tools T.
 3. Form the joint (augmented) distribution p_aug over V ∪ T via a single softmax
     over concatenated logits [W_V; W_T] h_{i-1}.
 4. Take a tentative next item x_{i+1}^{(0)} from p_aug (argmax if temperature==0).
     - If it is a word token (in V): accept it (no reranking) and continue.
     - If it is a tool token (in T):
          a. Build T_k: the top-k tools by their p_aug probability (ensure inclusion
              of the tentative tool if outside top-k).
          b. Build a reranker prompt listing those k tools (in descending base
              relevance) plus REJ.
          c. Run the reranker head to obtain p_rank over T_k ∪ {REJ} (softmax over
              only those candidates).
          d. If reranker chooses REJ: discard tentative tool and instead draw the
              next word from the base word distribution p_LLM (the word-only slice
              of token logits). Otherwise accept the chosen tool and enter an
              argument completion sub-generation which evaluates the tool, replaces
              the call with its numeric result, then resumes step (i+1).

Stopping conditions: regex match of final answer pattern or max_gen_len tokens
added (counting both words and substituted results).

Notes:
 - Only features explicitly required by the described algorithm are present.
 - No extra debug flags, reranker sampling temperature, or artificial limits on
    the number of tool calls are exposed (beyond overall max_gen_len).
 - Per-step logs are kept ("steps") for traceability; remove if undesired.
"""

import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional, Any

import fire
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from evaluation.metrics import parse_answer, accuracy
from funchub.math import *  # noqa: F401,F403  (used dynamically via eval in parse_and_eval)

# ------------------------------------------------------------------------------------
# Shared / reused components (lightly copied to avoid import cycles)
# ------------------------------------------------------------------------------------

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
    """Linear projection over hidden -> tool logits (frozen base)."""
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
        if isinstance(state, dict) and "func_head.weight" in state:
            self.load_state_dict(state, strict=False)
        else:
            self.func_head.load_state_dict(state, strict=True)
        self.func_head.to(next(self.model.parameters()).device)
        self.eval()

    @torch.inference_mode()
    def last_step_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        logits = self.func_head(h.float())  # (B,K)
        return logits


class RerankerHead(nn.Module):
    """Embedding-based reranker head (copied minimal logic from training)."""
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.W = nn.Embedding(num_classes, hidden_size)

    def forward_logits(self, h: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        # h: (B,H), cand_ids: (B,Kc)
        emb = self.W(cand_ids)  # (B,Kc,H)
        logits = torch.einsum("bd,bkd->bk", h, emb)
        return logits


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------

def read_templates(dataset: str, tdir: str) -> Dict[str, str]:
    tdir_full = f"data/{dataset}/{tdir}"
    templates: Dict[str, str] = {}
    for name in os.listdir(tdir_full):
        with open(os.path.join(tdir_full, name), "r", encoding="utf-8") as f:
            templates[name.split("_")[-1].replace(".txt", "")] = f.read()
    return templates


def load_func_dict(dataset: str) -> Dict[str, int]:
    with open(f"data/{dataset}/func_dict.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_questions(dataset: str, test_file: str) -> Tuple[List[str], List[float]]:
    with open(f"data/{dataset}/{test_file}", "r", encoding="utf-8") as f:
        data = json.load(f)
    qs, labels = [], []
    for item in data:
        qs.append(item["question"])
        labels.append(item.get("answer"))
    return qs, labels

def load_gold_tools(dataset: str, test_file: str) -> List[Optional[str]]:
    """Return list of gold tool names (e.g. '<add>') per question or None if unavailable.

    Expects each item to maybe contain a 'func' field like '<add>(1,2)=3'.
    We extract the substring starting with '<' up to and including the first '>'.
    """
    path = f"data/{dataset}/{test_file}"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tools: List[Optional[str]] = []
    for item in data:
        func_expr = item.get("func")
        if isinstance(func_expr, str) and func_expr.startswith("<") and ">" in func_expr:
            tools.append(func_expr.split(">",1)[0] + ">")
        else:
            tools.append(None)
    return tools


class AnswerStopper(StoppingCriteria):
    pattern = re.compile(r"(?:####|answer is)\s*\$?[-+]?[\d,]+(?:\.\d+)? .*?\.\n\n", re.IGNORECASE)
    def __init__(self, tok, start_idx: int):
        super().__init__()
        self.tok = tok
        self.start_idx = start_idx
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:  # type: ignore[override]
        generated_ids = input_ids[0][self.start_idx:]
        if len(generated_ids) == 0:
            return False
        text = self.tok.decode(generated_ids, skip_special_tokens=True)
        return bool(self.pattern.search(text))


def parse_and_eval(op: str, args_raw: str) -> Tuple[bool, str, float]:
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
        res = eval(f"{op[1:-1]}_{s}")  # noqa: S307 (controlled namespace)
        return True, s, float(res)
    except Exception:
        return False, s, 0.0


def build_prompt(prefix_text: str,
                 candidates: List[str],
                 tool_docs: Dict[str, str],
                 style: str = "plain",
                 rej_label: str = "REJ",
                 rej_doc: str = "continue the answer without calling any tool.") -> str:
    if style == "plain":
        lines = ["System:\nYou are a tool selector. Choose the next action from the options below.\n",
                 "Tools (in descending relevance from the base model):"]
        for i, t in enumerate(candidates, start=1):
            doc = tool_docs.get(t, f"{t}(...) usage.")
            lines.append(f"{i}) {t}: {doc}")
        lines.append(f"{len(candidates)+1}) {rej_label}: {rej_doc}")
        opts = " or ".join(candidates + [rej_label])
        lines.append("\nInstruction:\nGiven the context, choose EXACTLY ONE option from the list above.\n"
                     f"Reply with ONLY the option token EXACTLY as written: {opts}\n")
        lines.append("Context:")
        lines.append(prefix_text)
        lines.append("\nAnswer:\n")
        return "\n".join(lines)
    # default fallback style
    tool_lines = []
    for i, t in enumerate(candidates, start=1):
        doc = tool_docs.get(t, f"{t}(...) usage.")
        tool_lines.append(f"{i}) {t}: {doc}")
    tool_lines.append(f"{len(candidates)+1}) {rej_label}: {rej_doc}")
    opts = " or ".join(candidates + [rej_label])
    user = ("You are a tool selector. Choose the next action.\n" +
            "Tools (in descending relevance from the base model):\n" +
            "\n".join(tool_lines) +
            "\n\nInstruction: Choose EXACTLY ONE option from above. Reply with ONLY: " + opts +
            "\nContext:\n" + prefix_text + "\nAnswer:")
    return f"<s>[INST] {user} [/INST]"


# ------------------------------------------------------------------------------------
# Core Toolken+ inference algorithm implementation
# ------------------------------------------------------------------------------------

def load_reranker_head(reranker_head_path: str, device: torch.device):
    ckpt = torch.load(reranker_head_path, map_location="cpu")
    # Accept either wrapped dict with state_dict or raw
    state_dict = ckpt.get("state_dict", ckpt)
    toolname2idx = ckpt.get("toolname2idx")
    hidden_size = ckpt.get("hidden_size")
    style = ckpt.get("style", "plain")
    train_top_k_tools = ckpt.get("train_top_k_tools")  # may be None for older checkpoints
    if toolname2idx is None or hidden_size is None:
        raise ValueError("Reranker checkpoint missing toolname2idx/hidden_size")
    head = RerankerHead(int(hidden_size), len(toolname2idx))
    head.load_state_dict(state_dict, strict=True)
    head.to(device)
    head.eval()
    return head, toolname2idx, style, train_top_k_tools


class DigitsMaskOnce:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.digit_ids = set()
        for tid in range(tokenizer.vocab_size):
            try:
                txt = tokenizer.decode([tid], skip_special_tokens=True)
            except Exception:
                continue
            if len(txt) == 1 and txt.isdigit():
                self.digit_ids.add(tid)
        self.enable = False

    def apply(self, token_logits: torch.Tensor):
        if not self.enable:
            return token_logits
        if self.digit_ids:
            ids = list(self.digit_ids)
            token_logits[:, ids] = -1e9
        self.enable = False
        return token_logits


@torch.inference_mode()
def toolken_plus_infer_one(
    model,
    tok,
    question: str,
    templates: Dict[str, str],
    func_head: FunctionHeadOnlyLight,
    reranker_head: RerankerHead,
    toolname2idx: Dict[str, int],
    tool_docs: Dict[str, str],
    top_k_tools: int = 5,
    max_gen_len: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    style: str = "plain",
) -> Dict[str, Any]:
    """Run Toolken+ inference for a single question.

    Parameters
    ----------
    model : PreTrainedModel
        The base causal LM.
    tok : PreTrainedTokenizer
        Tokenizer for the model.
    question : str
        Input question string.
    templates : dict
        Prompt templates keyed by name (expects at least 'general').
    func_head : FunctionHeadOnlyLight
        Trained lightweight function/tool head (provides tool logits W_T h).
    reranker_head : RerankerHead
        Trained reranker embedding head.
    toolname2idx : dict
        Mapping of tool name -> global class id (includes 'REJ').
    tool_docs : dict
        Documentation snippets per tool for reranking prompt.
    top_k_tools : int
        Size of candidate set T_k for reranker (Algorithm 1).
    max_gen_len : int
        Maximum decoding iterations (upper bound on steps).
    temperature, top_p : float
        Standard sampling controls (argmax if temperature==0).
    style : str
        Reranker prompt style (e.g. 'plain').

    Returns
    -------
    dict with keys: generation, func_calls, steps, status.
    """

    func_map = list(func_head.func_dict.keys())  # e.g. ['<add>', '<subtract>', ...]
    # Guarantee REJ in reranker mapping
    assert "REJ" in toolname2idx, "Reranker mapping must include REJ"

    general_tmpl = templates.get("general", next(iter(templates.values())))
    func_tmpl_fallback = templates.get("func", general_tmpl)

    cur_visible = ""  # what we show to subsequent forwards
    func_calls: List[str] = []
    start_len: List[int] = []
    end_len: List[int] = []
    steps: List[Dict[str, Any]] = []

    digits_mask = DigitsMaskOnce(tok)
    # Stopping pattern regex reuse
    stop_regex = re.compile(r"(?:####|answer is)\s*\$?[-+]?[\d,]+(?:\.\d+)? .*?\.\n\n", re.IGNORECASE)

    def sample_from_logits(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature and temperature > 0:
            logits = logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            if 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > top_p
                if torch.any(cutoff):
                    first = int(torch.nonzero(cutoff, as_tuple=False)[0].item())
                    keep = sorted_idx[:first + 1]
                else:
                    keep = sorted_idx
                mask = torch.zeros_like(probs)
                mask[keep] = probs[keep]
                probs = mask / mask.sum().clamp(min=1e-9)
            choice = int(torch.multinomial(probs, 1).item())
        else:
            choice = int(torch.argmax(logits).item())
        return choice

    for step in range(max_gen_len):
        # 1) Build base prompt
        base_prompt = general_tmpl.replace("[QUESTION]", question) + cur_visible
        enc = tok(base_prompt, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        token_logits_last = out.logits[:, -1, :].float()  # (1,V)
        h_last = out.hidden_states[-1][:, -1, :]          # (1,H)

        # 2) Tool logits
        func_logits = func_head.func_head(h_last.float())  # (1,K)

        # 3) Build joint logits -> p_aug
        joint_logits = torch.cat([token_logits_last, func_logits], dim=-1)  # (1,V+K)
        joint_probs = torch.softmax(joint_logits, dim=-1)[0]
        V = token_logits_last.size(-1)
        K = func_logits.size(-1)

        # tentative pick
        if temperature > 0:
            choice = sample_from_logits(joint_logits[0], temperature, top_p)
        else:
            choice = int(torch.argmax(joint_logits[0]).item())

        step_log: Dict[str, Any] = {"step": step}

        if choice < V:  # word path
            tok_id = choice
            token_logits_last = digits_mask.apply(token_logits_last)
            token_text = tok.decode([tok_id], skip_special_tokens=True)
            if token_text == "":
                token_text = tok.decode([tok_id], skip_special_tokens=False)
            cur_visible += token_text
            step_log["tentative"] = {"type": "word", "token": token_text, "prob": float(joint_probs[choice])}
            step_log["committed"] = {"type": "word", "text": token_text}
            steps.append(step_log)
        else:  # tool tentative
            tool_idx = choice - V
            op = func_head.id2func.get(int(tool_idx), None)
            if op is None:
                # Fallback treat as no-op word (skip)
                continue
            step_log["tentative"] = {"type": "tool", "tool": op, "prob": float(joint_probs[choice])}

            # 4) Build top-k tool candidate set using tool segment probabilities
            tool_probs = joint_probs[V:]
            topk = min(top_k_tools, K)
            base_sorted = torch.argsort(tool_probs, descending=True)
            topk_indices = base_sorted[:topk].tolist()
            if tool_idx not in topk_indices:
                topk_indices.append(tool_idx)  # ensure inclusion
            # Order by base prob descending
            topk_indices = sorted(topk_indices, key=lambda i: float(tool_probs[i]), reverse=True)
            candidates_tools = [func_head.id2func[i] for i in topk_indices]

            step_log["tool_topk"] = [
                {"tool": func_head.id2func[i], "base_prob": float(tool_probs[i])} for i in topk_indices
            ]

            # 5) Reranker scoring over candidates + REJ
            prefix_text = base_prompt  # includes question + current visible reasoning
            rerank_prompt = build_prompt(prefix_text, candidates_tools, tool_docs, style=style, rej_label="REJ")
            r_enc = tok(rerank_prompt, return_tensors="pt").to(model.device)
            r_out = model(input_ids=r_enc["input_ids"], attention_mask=r_enc.get("attention_mask"), output_hidden_states=True)
            r_h_last = r_out.hidden_states[-1][:, -1, :]
            rerank_cands = candidates_tools + ["REJ"]
            cand_ids = torch.tensor([[toolname2idx[c] for c in rerank_cands]], device=model.device)
            rerank_logits = reranker_head.forward_logits(r_h_last.float(), cand_ids)
            # Temperature for reranker (separate)
            rr_logits = rerank_logits[0]
            rr_probs = torch.softmax(rr_logits, dim=-1)
            rr_choice = int(torch.argmax(rr_logits).item())
            chosen_label = rerank_cands[rr_choice]
            step_log["rerank"] = {
                "candidates": rerank_cands,
                "logits": [float(x) for x in rr_logits.tolist()],
                "probs": [float(x) for x in rr_probs.tolist()],
                "chosen": chosen_label,
            }

            if chosen_label == "REJ":
                # discard tentative tool -> sample from pure word distribution p_LLM
                p_llm_logits = token_logits_last[0]
                word_choice = sample_from_logits(p_llm_logits, temperature, top_p)
                word_text = tok.decode([word_choice], skip_special_tokens=True)
                if word_text == "":
                    word_text = tok.decode([word_choice], skip_special_tokens=False)
                cur_visible += word_text
                step_log["committed"] = {"type": "word", "text": word_text}
                steps.append(step_log)
                continue
            else:
                # Accept chosen tool
                accepted_op = chosen_label
                cur_visible += f"{accepted_op}("
                step_log["committed"] = {"type": "tool", "tool": accepted_op}
                steps.append(step_log)

                # Argument generation sub-phase
                # Insert previous evaluated calls for context replacement similar to original
                if start_len and end_len:
                    tmp = cur_visible
                    bias = 0
                    for i_call in range(len(start_len)):
                        tmp = tmp[: start_len[i_call] + bias] + func_calls[i_call] + tmp[end_len[i_call] + bias :]
                        bias += len(func_calls[i_call]) - (end_len[i_call] - start_len[i_call])
                    cur_with_funcs = tmp
                else:
                    cur_with_funcs = cur_visible

                sub_template = templates.get(accepted_op[1:-1], func_tmpl_fallback)
                sub_prompt = sub_template.replace("[QUESTION]", question) + cur_with_funcs
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
                after = cur_visible + sub_text
                tail = after.split(accepted_op, 1)[-1]
                args_part = tail.split(")", 1)[0] + ")"
                ok, args_norm, res = False, args_part, 0.0
                try:
                    ok, args_norm, res = parse_and_eval(accepted_op, args_part)
                except Exception:
                    ok = False
                if ok:
                    func_calls.append(f"{accepted_op}{args_norm} = {res}")
                    start_len.append(len(cur_visible.split(accepted_op)[0]))
                    # Replace the tool call with result
                    cur_visible = cur_visible.split(accepted_op)[0] + str(res)
                    end_len.append(len(cur_visible))
                    digits_mask.enable = True  # avoid immediate digit duplication
                    # Successful tool evaluation
                else:
                    # Keep the raw call text (truncate if huge)
                    raw_insert = f"{accepted_op}{args_part}"
                    cur_visible = cur_visible + args_part  # leave call explicit
                    func_calls.append(raw_insert + " = <ERR>")
                    start_len.append(len(cur_visible) - len(args_part))
                    end_len.append(len(cur_visible))
                    # Failed argument parse; raw call kept

        # Stopping conditions
        if stop_regex.search(cur_visible):
            break

    return {
        "generation": cur_visible.strip(),
        "func_calls": func_calls,
        "steps": steps,
        "status": "success",
    }


# ------------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------------
from tqdm import tqdm
def main(
    model_name_or_path: str = "/workspace/.hf_home/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
    dataset: str = "funcqa",
    test_file: str = "funcqa_oh.json",
    tdir: str = "template_oh",
    func_head_path: str = "/workspace/toolken/emre_code/checkpoints_head_only_plus/workspace/.hf_home/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d/head_best.pth",
    reranker_head_path: str = "reranker_head_best_best.pt",
    tool_docs_json: Optional[str] = None,
    top_k_tools: int = 3,
    max_gen_len: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.95,
    dtype: str = "bf16",
    output_dir: str = "outputs/funcqa",
    output_name: str = "inference-funcqa-reranker_half_split.jsonl",
    max_samples: Optional[int] = None,
    diagnostics: bool = True,
    diagnostics_output: Optional[str] = None,
):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not func_head_path or not reranker_head_path:
        raise ValueError("Both --func_head_path and --reranker_head_path are required for Toolken+ inference.")

    os.makedirs(output_dir, exist_ok=True)

    model, tok = load_gemma(model_name_or_path, device, dtype)

    # Load function head
    func_dict = load_func_dict(dataset)
    func_head = FunctionHeadOnlyLight(model, func_dict)
    func_head.load_weights(func_head_path)

    # Load reranker head and tool docs
    reranker_head, toolname2idx, style, train_top_k_from_ckpt = load_reranker_head(reranker_head_path, device)
    tool_docs: Dict[str, str] = {}
    if tool_docs_json and os.path.exists(tool_docs_json):
        with open(tool_docs_json, "r", encoding="utf-8") as f:
            tool_docs = json.load(f)
    # Minimal stubs
    for name in toolname2idx.keys():
        if name == "REJ":
            continue
        tool_docs.setdefault(name, f"{name}(...) invocation.")

    templates = read_templates(dataset, tdir)
    # If checkpoint encodes a training-time top-k, prefer it unless user explicitly overrides via CLI
    # Heuristic: if user left top_k_tools at default (3) but ckpt has different value, adopt ckpt value.
    if train_top_k_from_ckpt is not None and top_k_tools != train_top_k_from_ckpt:
        # Only override if user didn't explicitly set (cannot know; assume override always but emit warning)
        print(f"[INFO] Overriding provided top_k_tools={top_k_tools} with checkpoint train_top_k_tools={train_top_k_from_ckpt} to ensure consistency.")
        top_k_tools = int(train_top_k_from_ckpt)
    questions, labels = build_questions(dataset, test_file)
    gold_tools = load_gold_tools(dataset, test_file)
    if max_samples is not None:
        questions = questions[:max_samples]
        labels = labels[:max_samples]
        gold_tools = gold_tools[:max_samples] if gold_tools else []

    out_path = os.path.join(output_dir, model_name_or_path.replace("/", "_") + "-" + output_name)

    preds: List[float] = []
    # Diagnostics accumulators
    diag = {
        "questions": 0,
        "total_steps": 0,
        "reranker_invocations": 0,
        "tool_tentatives": 0,
        "tool_committed": 0,
        "tool_rejected": 0,
        "word_committed": 0,
        "label_accept_counts": {},   # accepted tool label -> count
        "rej_accept_count": 0,        # times reranker chose REJ
        "candidate_total": 0,
        "candidate_events": 0,
    # New: reranker agreement with initial function head tentative tool
    # Counts only when tentative was a tool and reranker chose a (non-REJ) tool.
    "rerank_tool_agree": 0,       # chosen tool == tentative tool
    "rerank_tool_decisions": 0,   # reranker chose a tool (not REJ)
    # Gold tool accuracy metrics (per question, first tool decision only)
    "func_head_gold_correct": 0,
    "func_head_gold_total": 0,
    "reranker_gold_correct": 0,
    "reranker_gold_total": 0,
    # Extended coverage metrics
    "questions_with_gold": 0,
    "head_first_predicted": 0,
    "head_first_correct": 0,
    "head_any_correct": 0,
    "head_no_proposal": 0,
    "head_missed_gold": 0,
    "head_spurious": 0,
    "reranker_first_predicted": 0,
    "reranker_first_correct": 0,
    "reranker_any_correct": 0,
    "reranker_missed_gold": 0,
    "reranker_spurious": 0,
    "flip_help": 0,
    "flip_hurt": 0,
    "total_tool_attempts_head": 0,
    "total_tool_accepts_reranker": 0,
    }
    n_step = max(len(questions) // 10, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (q, gold) in enumerate(tqdm(list(zip(questions, labels)), total=len(questions), desc="FuncInfer")):
                try:
                    log = toolken_plus_infer_one(
                        model=model,
                        tok=tok,
                        question=q,
                        templates=templates,
                        func_head=func_head,
                        reranker_head=reranker_head,
                        toolname2idx=toolname2idx,
                        tool_docs=tool_docs,
                        top_k_tools=top_k_tools,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                        style=style,
                    )
                    gen = log["generation"]
                    pv = parse_answer(gen, pattern="answer is")
                    preds.append(pv)
                    rec = {
                        "index": i,
                        "question": q,
                        "generation": gen,
                        "pred": pv,
                        "gold": gold,
                        "func_calls": log.get("func_calls", []),
                        "steps": log.get("steps", []),
                        "status": log.get("status", "success"),
                        "reranker_active": True,
                    }
                    f.write(json.dumps(rec) + "\n")
                except Exception as e:
                    preds.append(None)
                    rec = {
                        "index": i,
                        "question": q,
                        "generation": "",
                        "pred": None,
                        "gold": gold,
                        "func_calls": [],
                        "steps": [],
                        "status": f"error: {e}",
                        "reranker_active": True,
                    }
                    f.write(json.dumps(rec) + "\n")
                # Diagnostics aggregation per question
                if diagnostics:
                    diag["questions"] += 1
                    steps = rec.get("steps", [])
                    # Track if we've already evaluated gold tool accuracy for this question
                    gold_tool_recorded = False
                    head_tools = []
                    reranker_tools = []
                    for st in steps:
                        diag["total_steps"] += 1
                        tent = st.get("tentative", {})
                        comm = st.get("committed", {})
                        if tent.get("type") == "tool":
                            diag["tool_tentatives"] += 1
                            if tent.get("tool"):
                                head_tools.append(tent.get("tool"))
                        if comm.get("type") == "tool":
                            diag["tool_committed"] += 1
                            tool_name = comm.get("tool")
                            if tool_name:
                                diag["label_accept_counts"][tool_name] = diag["label_accept_counts"].get(tool_name, 0) + 1
                                reranker_tools.append(tool_name)
                        elif comm.get("type") == "word":
                            diag["word_committed"] += 1
                        rr = st.get("rerank")
                        if rr:
                            diag["reranker_invocations"] += 1
                            diag["candidate_total"] += len(rr.get("candidates", []))
                            diag["candidate_events"] += 1
                            chosen = rr.get("chosen")
                            if chosen == "REJ":
                                diag["rej_accept_count"] += 1
                                # If tentative was tool and REJ chosen, count rejection
                                if tent.get("type") == "tool":
                                    diag["tool_rejected"] += 1
                            else:
                                # Reranker selected a tool (not REJ). Track agreement with tentative tool.
                                if tent.get("type") == "tool":
                                    diag["rerank_tool_decisions"] += 1
                                    if chosen == tent.get("tool"):
                                        diag["rerank_tool_agree"] += 1
                            # Gold tool accuracy (only first tool decision for this question)
                            if (not gold_tool_recorded) and tent.get("type") == "tool":
                                gold_tool = gold_tools[i] if i < len(gold_tools) else None
                                if gold_tool:
                                    diag["func_head_gold_total"] += 1
                                    if tent.get("tool") == gold_tool:
                                        diag["func_head_gold_correct"] += 1
                                    if chosen != "REJ":
                                        diag["reranker_gold_total"] += 1
                                        if chosen == gold_tool:
                                            diag["reranker_gold_correct"] += 1
                                gold_tool_recorded = True
                        # If there is no rerank object (word committed) skip gold metrics
                    # Edge case: If there was a tentative tool but no rerank object? (Shouldn't happen)
                    gold_tool = gold_tools[i] if i < len(gold_tools) else None
                    gold_exists = gold_tool is not None
                    if gold_exists:
                        diag["questions_with_gold"] += 1
                    # Head metrics
                    if head_tools:
                        diag["head_first_predicted"] += 1
                        diag["total_tool_attempts_head"] += len(head_tools)
                        if gold_exists and head_tools[0] == gold_tool:
                            diag["head_first_correct"] += 1
                        if gold_exists and gold_tool in head_tools:
                            diag["head_any_correct"] += 1
                        if gold_exists and gold_tool not in head_tools:
                            diag["head_missed_gold"] += 1
                    else:
                        if gold_exists:
                            diag["head_no_proposal"] += 1
                            diag["head_missed_gold"] += 1
                    if (not gold_exists) and head_tools:
                        diag["head_spurious"] += 1
                    # Reranker accepted tools metrics
                    if reranker_tools:
                        diag["reranker_first_predicted"] += 1
                        diag["total_tool_accepts_reranker"] += len(reranker_tools)
                        if gold_exists and reranker_tools[0] == gold_tool:
                            diag["reranker_first_correct"] += 1
                        if gold_exists and gold_tool in reranker_tools:
                            diag["reranker_any_correct"] += 1
                        if gold_exists and gold_tool not in reranker_tools:
                            diag["reranker_missed_gold"] += 1
                    else:
                        if gold_exists:
                            diag["reranker_missed_gold"] += 1
                    if (not gold_exists) and reranker_tools:
                        diag["reranker_spurious"] += 1
                    # Flip diagnostics (first tool perspective)
                    if gold_exists and head_tools:
                        head_first = head_tools[0]
                        rer_first = reranker_tools[0] if reranker_tools else None
                        if head_first != gold_tool and rer_first == gold_tool:
                            diag["flip_help"] += 1
                        if head_first == gold_tool and (rer_first is None or rer_first != gold_tool):
                            diag["flip_hurt"] += 1
                
                # Print intermediate metrics every n_step samples
                if (i + 1) % n_step == 0:
                    curr_em = accuracy(preds, labels[:i+1], type="em") 
                    curr_approx = accuracy(preds, labels[:i+1], type="approx")
                    curr_dur = time.time() - t0
                    if diagnostics and diag["total_steps"] > 0:
                        rerank_rate = diag["reranker_invocations"] / max(diag["total_steps"], 1)
                        tool_accept_rate = diag["tool_committed"] / max(diag["tool_tentatives"], 1) if diag["tool_tentatives"] else 0.0
                        agree_rate = diag["rerank_tool_agree"] / max(diag["rerank_tool_decisions"], 1) if diag["rerank_tool_decisions"] else 0.0
                        head_tool_acc = diag["func_head_gold_correct"] / max(diag["func_head_gold_total"], 1) if diag["func_head_gold_total"] else 0.0
                        rerank_tool_acc = diag["reranker_gold_correct"] / max(diag["reranker_gold_total"], 1) if diag["reranker_gold_total"] else 0.0
                        print(f"[Step {i+1}/{len(labels)}] EM={curr_em:.4f} | approx={curr_approx:.4f} | time={curr_dur:.1f}s | rerank_rate={rerank_rate:.3f} | tool_tent={diag['tool_tentatives']} | tool_accpt={diag['tool_committed']} | tool_accept_rate={tool_accept_rate:.3f} | agree_rate={agree_rate:.3f} | head_tool_acc={head_tool_acc:.3f} | rerank_tool_acc={rerank_tool_acc:.3f} | REJ={diag['rej_accept_count']}")
                    else:
                        print(f"[Step {i+1}/{len(labels)}] EM={curr_em:.4f} | approx={curr_approx:.4f} | time={curr_dur:.1f}s")

    em = accuracy(preds, labels, type="em")
    approx = accuracy(preds, labels, type="approx")
    dur = time.time() - t0
    print(f"[DONE] Saved outputs to {out_path} | EM={em:.4f} | approx={approx:.4f} | n={len(labels)} | time={dur:.1f}s")
    if diagnostics and diag["questions"] > 0:
        avg_cand = (diag["candidate_total"] / max(diag["candidate_events"], 1)) if diag["candidate_events"] else 0.0
        tool_accept_rate = diag["tool_committed"] / max(diag["tool_tentatives"], 1)
        rerank_inv_rate = diag["reranker_invocations"] / max(diag["total_steps"], 1)
        print("[DIAG] Inference diagnostics summary:")
        print(f"  questions: {diag['questions']}")
        print(f"  total_steps: {diag['total_steps']}")
        print(f"  reranker_invocations: {diag['reranker_invocations']} (rate={rerank_inv_rate:.3f})")
        print(f"  tool_tentatives: {diag['tool_tentatives']}")
        print(f"  tool_committed: {diag['tool_committed']} (accept_rate={tool_accept_rate:.3f})")
        print(f"  tool_rejected: {diag['tool_rejected']}")
        print(f"  word_committed: {diag['word_committed']}")
        print(f"  reranker_REJ_decisions: {diag['rej_accept_count']}")
        print(f"  avg_candidate_count: {avg_cand:.2f}")
        if diag['rerank_tool_decisions']:
            agree_rate = diag['rerank_tool_agree'] / max(diag['rerank_tool_decisions'], 1)
            print(f"  rerank_tool_decisions: {diag['rerank_tool_decisions']} (agree={diag['rerank_tool_agree']}, agree_rate={agree_rate:.3f})")
        if diag['func_head_gold_total']:
            head_acc = diag['func_head_gold_correct'] / max(diag['func_head_gold_total'], 1)
            print(f"  func_head_gold: correct={diag['func_head_gold_correct']} total={diag['func_head_gold_total']} acc={head_acc:.3f}")
        if diag['reranker_gold_total']:
            rerank_acc = diag['reranker_gold_correct'] / max(diag['reranker_gold_total'], 1)
            delta = rerank_acc - (diag['func_head_gold_correct'] / max(diag['func_head_gold_total'],1) if diag['func_head_gold_total'] else 0.0)
            print(f"  reranker_gold: correct={diag['reranker_gold_correct']} total={diag['reranker_gold_total']} acc={rerank_acc:.3f} (delta_vs_head={delta:+.3f})")
        if diag['label_accept_counts']:
            top_labels = sorted(diag['label_accept_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"  top_accepted_tools: {top_labels}")
        # Extended metrics
        q = diag['questions']
        gold_q = diag['questions_with_gold']
        no_gold_q = q - gold_q
        head_first_precision = diag['head_first_correct'] / max(diag['head_first_predicted'], 1)
        head_any_recall = diag['head_any_correct'] / max(gold_q, 1)
        head_missed_rate = diag['head_missed_gold'] / max(gold_q, 1)
        head_no_prop_rate = diag['head_no_proposal'] / max(gold_q, 1)
        head_spurious_rate = diag['head_spurious'] / max(no_gold_q, 1) if no_gold_q > 0 else 0.0
        rer_first_precision = diag['reranker_first_correct'] / max(diag['reranker_first_predicted'], 1)
        rer_any_recall = diag['reranker_any_correct'] / max(gold_q, 1)
        rer_missed_rate = diag['reranker_missed_gold'] / max(gold_q, 1)
        rer_spurious_rate = diag['reranker_spurious'] / max(no_gold_q, 1) if no_gold_q > 0 else 0.0
        flip_help_rate = diag['flip_help'] / max(gold_q, 1)
        flip_hurt_rate = diag['flip_hurt'] / max(gold_q, 1)
        avg_attempts_head = diag['total_tool_attempts_head'] / max(q,1)
        avg_accepts_rer = diag['total_tool_accepts_reranker'] / max(q,1)
        print("  -- Extended head/reranker metrics --")
        print(f"  head_first_precision={head_first_precision:.3f} | head_any_recall={head_any_recall:.3f} | head_missed_rate={head_missed_rate:.3f} | head_no_proposal_rate={head_no_prop_rate:.3f} | head_spurious_rate={head_spurious_rate:.3f}")
        print(f"  reranker_first_precision={rer_first_precision:.3f} | reranker_any_recall={rer_any_recall:.3f} | reranker_missed_rate={rer_missed_rate:.3f} | reranker_spurious_rate={rer_spurious_rate:.3f}")
        print(f"  flip_help={diag['flip_help']} (rate={flip_help_rate:.3f}) | flip_hurt={diag['flip_hurt']} (rate={flip_hurt_rate:.3f})")
        print(f"  avg_tool_attempts_head={avg_attempts_head:.3f} | avg_tool_accepts_reranker={avg_accepts_rer:.3f}")
        # Optionally write json
        if diagnostics_output:
            diag_out_path = os.path.join(output_dir, diagnostics_output)
            try:
                diag_ext = {
                    "em": em,
                    "approx": approx,
                    "head_first_precision": head_first_precision,
                    "head_any_recall": head_any_recall,
                    "head_missed_rate": head_missed_rate,
                    "head_no_proposal_rate": head_no_prop_rate,
                    "head_spurious_rate": head_spurious_rate,
                    "reranker_first_precision": rer_first_precision,
                    "reranker_any_recall": rer_any_recall,
                    "reranker_missed_rate": rer_missed_rate,
                    "reranker_spurious_rate": rer_spurious_rate,
                    "flip_help_rate": flip_help_rate,
                    "flip_hurt_rate": flip_hurt_rate,
                    "avg_tool_attempts_head": avg_attempts_head,
                    "avg_tool_accepts_reranker": avg_accepts_rer,
                }
                with open(diag_out_path, "w", encoding="utf-8") as f:
                    json.dump({**diag, **diag_ext}, f, ensure_ascii=False, indent=2)
                print(f"[DIAG] Wrote diagnostics JSON to {diag_out_path}")
            except Exception as e:
                print(f"[DIAG][WARN] Could not write diagnostics file: {e}")


if __name__ == "__main__":
    fire.Fire(main)
