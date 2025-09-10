import re
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 1. Load Dataset ---
ds = {}
ds['train'] = load_dataset("json", data_files={"train": "data/funcqa/training_data/*.jsonl"}, split="train")

# --- 2. Initialize Tokenizer ---
MODEL = "meta-llama/Llama-3.2-1B" 
MODEL = "meta-llama/Llama-3.2-3B" 
MODEL =  "HuggingFaceTB/SmolLM2-1.7B"
MODEL = "google/gemma-3-4b-pt"
MODEL = "meta-llama/Meta-Llama-3-8B"
MODEL = "Qwen/Qwen3-8B"
MODEL = "google/gemma-3-12b-pt"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

# --- 3. A flexible regex to find the entire tool-call syntax block ---
pattern_for_removal = re.compile(r'<[a-z]+>\([^)]+\)=[\-\.0-9]+(?:<eoe>[\-\.0-9]+)?')

# --- 4. Process the Dataset with the New Robust Loop ---
mismatches = 0
processed_samples = []

for sample in tqdm(ds['train']):
    full_text = f"Q: {sample['question']}\nA: {sample['answer']}"
    matches = list(pattern_for_removal.finditer(full_text))

    if not matches:
        continue

    clean_text_parts = []
    char_spans = []
    target_equations = []
    target_numbers = []
    last_idx = 0

    for match in matches:
        clean_text_parts.append(full_text[last_idx:match.start()])
        syntax_block = match.group(0)
        core_match = re.search(r'(<[a-z]+>\([^)]+\)=([\-\.0-9]+))', syntax_block)
        
        if not core_match:
            last_idx = match.end()
            continue

        core_call_with_result = core_match.group(1)
        result_num = core_match.group(2)

        target_equations.append(core_call_with_result + "<eoe>")
        target_numbers.append(result_num)

        current_len = len("".join(clean_text_parts))
        char_start = current_len
        char_end = current_len + len(result_num)
        char_spans.append((char_start, char_end))
        
        clean_text_parts.append(result_num)
        last_idx = match.end()

    clean_text_parts.append(full_text[last_idx:])
    clean_text = "".join(clean_text_parts)

    encoding = tokenizer(clean_text, return_offsets_mapping=True)
    token_offsets = encoding["offset_mapping"]
    
    start_indices = []
    end_indices = []

    for char_start, char_end in char_spans:
        token_start_index = -1
        token_end_index = -1

        # --- START OF THE FINAL BUG FIX ---
        # Find the FIRST token that contains the start of the number
        for i, (offset_start, offset_end) in enumerate(token_offsets):
            if offset_start <= char_start < offset_end:
                token_start_index = i
                break # Stop once we find the first token

        # If we found a start, search from that point for the end token
        if token_start_index != -1:
            for i in range(token_start_index, len(token_offsets)):
                offset_start, offset_end = token_offsets[i]
                if offset_start <= (char_end - 1) < offset_end:
                    token_end_index = i
        # --- END OF THE FINAL BUG FIX ---
        
        if token_start_index != -1 and token_end_index != -1:
            start_indices.append(token_start_index)
            end_indices.append(token_end_index + 1)
        else:
            mismatches += 1

    if matches and start_indices:
        processed_samples.append({
            "text": clean_text,
            "start_token_idx": start_indices,
            "end_token_idx": end_indices,
            "tar_eq": target_equations,
            "tar_number": target_numbers,
        })

total_attempts = sum(len(list(pattern_for_removal.finditer(f"Q: {s['question']}\nA: {s['answer']}"))) for s in ds['train'])
mismatch_ratio = mismatches / total_attempts if total_attempts > 0 else 0
print(f"\nInitial processing mismatch ratio (unmappable tokens): {mismatch_ratio}")

# --- 5. Strict Validation ---
strict_total_val = 0
strict_ok = 0
mismatches_after_correction = []

for i, samp in enumerate(processed_samples):
    token_ids = tokenizer.encode(samp["text"])
    
    for j, (s, t, num) in enumerate(zip(samp["start_token_idx"], samp["end_token_idx"], samp.get("tar_number", []))):
        strict_total_val += 1
        span_text = tokenizer.decode(token_ids[s:t]).strip()
        num_stripped = num.strip()

        if span_text == num_stripped:
            strict_ok += 1
        elif span_text == f"0{num_stripped}":
             strict_ok += 1
        else:
            mismatches_after_correction.append({
                "sample_idx": i,
                "tool_call_idx": j, "s": s, "t": t, "expected": num, "span_text": span_text,
                "ctx_left": tokenizer.decode(token_ids[max(0, s-5):s]),
                "ctx_right": tokenizer.decode(token_ids[t:min(len(token_ids), t+5)]),
            })

print(f"Strict — total: {strict_total_val}, matches: {strict_ok}, mismatches: {len(mismatches_after_correction)}")

if mismatches_after_correction:
    print("\n--- Showing a few remaining mismatches for inspection ---")
    for r in mismatches_after_correction[:10]:
        print("-"*50)
        print({k: r[k] for k in ["sample_idx", "tool_call_idx", "s", "t", "expected", "span_text"]})
        print("L:", r["ctx_left"])
        print("R:", r["ctx_right"])

# --- 6. Save Processed Data and Function Dictionary ---
os.makedirs("inputs/funcqa", exist_ok=True)
output_path = os.path.join("inputs", "funcqa", MODEL.replace('/', '_') + "_processed.json")
with open(output_path, 'w') as f:
    json.dump(processed_samples, f, indent=4)
print(f"\n✅ Processed data saved to {output_path}")

with open("data/funcqa/func_dict.json", 'r') as f:
    func_dict = json.load(f)
func_dict_path = os.path.join("inputs", "funcqa", MODEL.replace('/', '_') + "_func_dict.json")
with open(func_dict_path, 'w') as f:
    json.dump(func_dict, f, indent=4)
print(f"✅ Function dictionary saved to {func_dict_path}")