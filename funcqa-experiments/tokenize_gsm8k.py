from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")


MODEL = "meta-llama/Meta-Llama-3-8B"
MODEL = "meta-llama/Llama-3.2-1B"
MODEL = "Qwen/Qwen3-8B"
MODEL = "mistralai/Mistral-7B-v0.3"
MODEL = "HuggingFaceTB/SmolLM2-1.7B"
MODEL = "microsoft/phi-4"

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

import re
tool_call_pattern = re.compile(r'(<<[^>]+>>)([^<]*)')

def parse_tool_call(expression: str):
    """
    Parses a mathematical expression and converts it into the ToolkenGPT format.
    Examples: 
    "48/2=24" -> "<divide>(48, 2)=24"
    "+30+46+38+11+18=143" -> "<add>(30, 46, 38, 11, 18)=143"
    """
    # Simple mapping from operator to function name
    op_to_func = {
        '+': 'add',
        '-': 'subtract',
        '*': 'multiply',
        '/': 'divide'
    }

    # Find the operator and split on equals sign
    if '=' in expression:
        expr_part, result = expression.split('=', 1)
        result = result.strip()
        
        # Handle multiple additions/subtractions
        for op, func_name in op_to_func.items():
            if op in expr_part:
                # Skip if it's just a leading sign
                if expr_part.strip() == op:
                    continue
                    
                # Split on operator, filter out empty strings
                parts = [p.strip() for p in expr_part.split(op) if p.strip()]
                
                # If we found valid parts, convert to function call format
                if parts:
                    return f"<{func_name}>({', '.join(parts)})={result}"
    
    return expression






from tqdm import tqdm
verbose = 1
mismatches = 0


processed_samples = []


for sample in tqdm(ds['train']):
    start_indices = []
    end_indices = []
    target_equations = []
    target_numbers = []
        
    removed_chars_offset = 0

    full_text = sample['question'] + " Let's think step by step. " + sample['answer']

    matches = list(tool_call_pattern.finditer(full_text))

    # Create the final clean text by removing only the <<...>> syntax.
    clean_text = re.sub(r'<<[^>]+>>', '', full_text)

    start_indices = []
    end_indices = []
    target_equations = []
    target_numbers = []

    removed_chars_offset = 0

    for match in matches:
        expression_part = match.group(1) # The <<...>> part
    
        expression = expression_part[2:-2] # The content inside
        expression = re.sub(r'(?<=[\s=+\-*/])\.(\d+)', r'0.\1', expression)  # Add 0 before decimal points

        following_text = match.group(2) # The text after <<...>>
        
        # The pattern should handle negative numbers as well.
        number_pattern = re.compile(r'-?[\d,]*\.?\d+')
        num_match = number_pattern.search(following_text)
        num = num_match.group(0) if num_match else None
        expected_num_str = expression.split('=')[-1].strip()

        """
        try:
            assert num == expected_num_str
        except AssertionError:
            mismatches += 1
            print("-"*25)
            print(num_match)
            print(f"Full text: {sample['answer']}")
            print("/"*10)
            print(f"Found match: {match.group(0)}")
            print(f"Expression part: {expression_part}")
            print(f"Following text: {following_text}")
            print(f"Number found: {num}")
            print(f"Expected number from expression: {expected_num_str}")
        """
        # 1. Calculate the character position of where the number *starts* in the clean_text.
        char_pos_start = (match.start() - removed_chars_offset) + num_match.start()

        # 2. Tokenize the clean text *before* the number's position to find the start_idx.
        text_before_result = clean_text[:char_pos_start]
        tokens_before = tokenizer.encode(text_before_result)
        start_idx = len(tokens_before)

        # 3. Calculate the character position of where the number *ends* in the clean_text.
        char_pos_end = char_pos_start + len(num)
        
        # 4. Tokenize the clean text up to the *end* of the number. The length of this
        #    token sequence is our end_idx.
        text_up_to_end_of_result = clean_text[:char_pos_end]
        tokens_up_to_end = tokenizer.encode(text_up_to_end_of_result)
        end_idx = len(tokens_up_to_end)

        # 5. Store the start and end indices for the current match.
        start_indices.append(start_idx-1)
        end_indices.append(end_idx)
        target_equations.append(parse_tool_call(expression))

        target_numbers.append(num)

        # Update the offset for the next iteration by adding the length of the
        # <<...>> syntax string we just processed.
        removed_chars_offset += len(expression_part)

        if num != expected_num_str and False:
            print(f"Text: {clean_text}")
            print(f"Before: {text_before_result}")
            print(f"After: {text_up_to_end_of_result}")
            print(f"Target equations: {target_equations}")
    
    processed_samples.append({
            "text": clean_text,
            "start_token_idx": start_indices,
            "end_token_idx": end_indices,
            "tar_eq": target_equations,
            "tar_number": target_numbers,
        })

print(f"Mismatch ratio: {mismatches/len(ds['train'])}")


# Strict validation: update samples in-place with corrected indices and match status.
strict_total = 0
strict_ok = 0
mismatches_after_correction = []

for i, samp in enumerate(processed_samples):
    # Initialize a new list to store the match status for each tool call in the sample
    samp["strict_match"] = []
    
    # keep tokenizer behavior consistent with above preprocessing (same encode)
    token_ids = tokenizer.encode(samp["text"])
    
    # Enumerate to get the index 'j' for updating the lists
    for j, (s, t, num) in enumerate(zip(samp["start_token_idx"], samp["end_token_idx"], samp.get("tar_number", []))):
        strict_total += 1
        span_text = tokenizer.decode(token_ids[s:t])
        span_text_m1 = tokenizer.decode(token_ids[s-1:t]).strip()
        span_text_p1 = tokenizer.decode(token_ids[s+1:t]).strip()

        is_match = False

        if (span_text == num):
            strict_ok += 1
            is_match = True
        # Check for a match with a left-shifted start index
        elif (span_text_m1 == num):
            strict_ok += 1
            is_match = True
            # Correct the start index in the sample
            samp["start_token_idx"][j] -= 1
        # Check for a match with a right-shifted start index
        elif (span_text_p1 == num):
            strict_ok += 1
            is_match = True
            # Correct the start index in the sample
            samp["start_token_idx"][j] += 1
        
        # Append the match status (True or False) for the current tool call
        samp["strict_match"].append(is_match)

        if not is_match:
            mismatches_after_correction.append({
                "sample_idx": i,
                "tool_call_idx": j,
                "s": s,
                "t": t,
                "expected": num,
                "span_text": span_text,
                "ctx_left": tokenizer.decode(token_ids[max(0, s-5):s]),
                "ctx_right": tokenizer.decode(token_ids[t:min(len(token_ids), t+5)]),
            })

print(f"Strict — total: {strict_total}, matches (with correction): {strict_ok}, mismatches: {len(mismatches_after_correction)}")

# Show a few remaining mismatches for inspection
for r in mismatches_after_correction[:10]:
    print("-"*50)
    print({k: r[k] for k in ["sample_idx", "tool_call_idx", "s", "t", "expected", "span_text"]})
    print("L:", r["ctx_left"])
    print("R:", r["ctx_right"])



import os
import json

# --- Save the processed data ---
os.makedirs("inputs", exist_ok=True)

output_path = os.path.join("inputs", MODEL.replace('/', '_') + "_processed.json")
with open(output_path, 'w') as f:
    json.dump(processed_samples, f, indent=4)
print(f"✅ Processed data saved to {output_path}")

# --- Create and save the final function dictionary ---
func_dict = {
    "<add>": 0,
    "<subtract>": 1,
    "<multiply>": 2,
    "<divide>": 3
}
func_dict_path = os.path.join("inputs", MODEL.replace('/', '_') + "_func_dict.json")
with open(func_dict_path, 'w') as f:
    json.dump(func_dict, f, indent=4)
print(f"✅ Function dictionary saved to {func_dict_path}")
