#!/usr/bin/env python3
"""
LLM-based mathematical expression extraction for GSM8K-XL dataset
Uses Qwen2.5-7B-Instruct with VLLM for fast inference
"""

import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re


class LLMMathExtractor:
    """Uses LLM with VLLM for fast mathematical expression extraction"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading model with VLLM: {model_name}")
        
        # Load model with VLLM for fast inference
        self.model = LLM(
            model=model_name,
            dtype="float16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True
        )
        
        # Load tokenizer separately for token mapping
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up sampling parameters for deterministic output
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50,
            stop=["\n", "###"],
            top_p=0.9
        )
            
        print(f"VLLM model loaded successfully! Much faster inference!")
    
    def create_extraction_prompt(self, text: str, target_equation: str) -> str:
        """
        Create a prompt for the LLM to extract mathematical expressions
        
        Args:
            text: The full text content
            target_equation: The target equation like "<add>(17, 16)=33<eoe>"
        
        Returns:
            prompt: Formatted prompt for the LLM
        """
        
        # Parse the target equation to understand what we're looking for
        function_match = re.search(r'<(\w+)>\((.*?)\)=(.*?)<eoe>', target_equation)
        if not function_match:
            return None
            
        func_name = function_match.group(1)
        args = function_match.group(2)
        result = function_match.group(3)
        
        # Map function names to operations
        op_map = {
            'add': 'addition',
            'subtract': 'subtraction', 
            'multiply': 'multiplication',
            'divide': 'division'
        }
        
        operation = op_map.get(func_name, func_name)
        
        prompt = f"""## Task
Your task is to find the exact substring in the provided text that represents a specific mathematical calculation. You will be given the text, the numbers, the operation, and the expected result.

## Inputs You Will Receive
1.  **TEXT_TO_SEARCH**: The block of text to search within.
2.  **OPERATION**: The type of math (e.g., 'addition', 'multiplication').
3.  **NUMBERS**: The list of numbers used in the operation (e.g., '144, 5'). 

## Step-by-Step Instructions
1.  **Identify Numbers**: Locate the core numbers from **NUMBERS** in the **TEXT_TO_SEARCH**. They may have different formatting (e.g., `$144` instead of `144`, or `.5` instead of `0.5`).
2.  **Scan for Operator**: Look for a mathematical operator (+, -, *, x, ×, /, ÷) connecting these numbers. Remember that multiplication can sometimes be implied by just a space.
3.  **Verify Calculation**: Check if the potential expression you found mathematically equals the **RESULT**.
4.  **Extract Verbatim**: If the expression is correct, extract the **exact, character-for-character substring** from the text. This is the most important step.
5.  **Handle Failure**: If you cannot find an exact expression that validates to the result, output the single string `NOT_FOUND`.

## Critical Rules & Constraints
- **Verbatim Extraction is Key**: You must return the expression **exactly** as it appears in the text. Do not add, remove, or reformat anything.
- **No Equals Sign**: Do NOT include the equals sign (`=`) or the result in your output.
- **Check Number Order**: The numbers might appear in a different order (e.g., `5 * 144` for numbers `144, 5`).
- **Watch for Symbols**: Pay close attention to currency symbols (`$`), decimal points (`.`), and varied multiplication symbols (`*`, `x`, `×`).
- **Final Output Format**: Your entire response must be **ONLY** the found expression or the text `NOT_FOUND`. Do not add explanations, apologies, or any other text.

## Examples

### Example 1: Multiplication with 'x'
- **TEXT_TO_SEARCH**: "The room dimensions are 8x12 feet, totaling 96 sq ft."
- **OPERATION**: multiplication
- **NUMBERS**: 8, 12
- **Correct Output**: `8x12`

### Example 2: Multiplication with Currency and Spaces
- **TEXT_TO_SEARCH**: "The total cost is 5 * $144 for the parts, which comes to $720."
- **OPERATION**: multiplication
- **NUMBERS**: 144, 5
- **Correct Output**: `5 * $144`

### Example 3: Subtraction with Multiple Terms
- **TEXT_TO_SEARCH**: "We started with 100 units, then used 50, then 30, then 15. The calculation 100 - 50 - 30 - 15 shows we have 5 left."
- **OPERATION**: subtraction
- **NUMBERS**: 100, 50, 30, 15
- **Correct Output**: `100 - 50 - 30 - 15`

### Example 4: Expression Not Found
- **TEXT_TO_SEARCH**: "He had ten dollars and spent five."
- **OPERATION**: subtraction
- **NUMBERS**: 10, 5
- **RESULT**: 5
- **Correct Output**: `NOT_FOUND`

---
*Begin Task*

**TEXT_TO_SEARCH**: {text}
**OPERATION**: {operation}
**NUMBERS**: {args}
**Correct Output**:
"""

        return prompt
    
    def extract_math_expression(self, text: str, target_equation: str) -> str:
        """
        Extract mathematical expression using VLLM
        
        Args:
            text: Full text content
            target_equation: Target equation like "<add>(17, 16)=33<eoe>"
        
        Returns:
            extracted_expression: The mathematical expression as it appears in text
        """
        prompt = self.create_extraction_prompt(text, target_equation)
        if not prompt:
            return None
            
        # Generate with VLLM (much faster!)
        outputs = self.model.generate([prompt], self.sampling_params)
        
        # Extract the response
        if outputs and len(outputs) > 0:
            full_response = outputs[0].outputs[0].text.strip()
            
            # Clean up the answer - remove common LLM explanation patterns
            answer = full_response.split('\n')[0].strip()  # Take first line only
            answer = answer.replace('"', '').replace("'", "")  # Remove quotes
                
            return answer
        
        return None
    
    def find_expression_in_tokens(self, text: str, math_expr: str, tokenizer) -> tuple:
        """
        Find token indices where math_expr appears in tokenized text
        
        Args:
            text: Full text content
            math_expr: Mathematical expression extracted by LLM
            tokenizer: The target tokenizer (Qwen for retokenization)
        
        Returns:
            (start_idx, end_idx): Token indices covering the expression
        """
        if not math_expr:
            return None, None
            
        # Find character position of expression in text
        char_start = text.find(math_expr)
        if char_start == -1:
            print(f"Warning: LLM extracted '{math_expr}' but it's not found in text")
            return None, None
            
        char_end = char_start + len(math_expr)
        
        # Tokenize text and map character positions to token positions
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Decode tokens one by one to find character mappings
        current_text = ""
        token_start_idx = None
        token_end_idx = None
        
        for i, token_id in enumerate(tokens):
            token_text = tokenizer.decode([token_id])
            token_char_start = len(current_text)
            current_text += token_text
            token_char_end = len(current_text)
            
            # Check if this token overlaps with our target character span
            if token_start_idx is None and token_char_start <= char_start < token_char_end:
                token_start_idx = i
            
            if token_start_idx is not None and char_end <= token_char_end:
                token_end_idx = i + 1
                break
        
        if token_start_idx is None or token_end_idx is None:
            print(f"Warning: Could not map '{math_expr}' to token indices")
            return None, None
            
        return token_start_idx, token_end_idx


def llm_retokenize_dataset(input_file: str, output_file: str, batch_size: int = 32):
    """
    Retokenize dataset using LLM extraction with batching for speed
    
    Args:
        input_file: Path to original dataset  
        output_file: Path to save retokenized dataset
        batch_size: Number of samples to process in parallel
    """
    print("🤖 Starting VLLM-based Mathematical Expression Extraction")
    print("=" * 60)
    
    # Initialize LLM extractor
    extractor = LLMMathExtractor()
    
    # Initialize target tokenizer (Qwen for training)
    target_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    # Load dataset
    print(f"Loading dataset from {input_file}")
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"Original dataset size: {len(original_data)}")
    
    # Process samples in batches for speed
    retokenized_data = []
    skipped_samples = 0
    
    # Create extraction tasks with proper sample mapping
    extraction_tasks = []  # Each task: {'sample_idx': i, 'eq_idx': j, 'prompt': prompt}
    
    for i, sample in enumerate(original_data):
        text = sample['text']
        target_equations = sample['tar_eq']
        
        for j, tar_eq in enumerate(target_equations):
            prompt = extractor.create_extraction_prompt(text, tar_eq)
            if prompt:
                extraction_tasks.append({
                    'sample_idx': i,
                    'eq_idx': j,
                    'prompt': prompt,
                    'text': text,
                    'tar_eq': tar_eq
                })
    
    print(f"Processing {len(extraction_tasks)} extraction tasks in batches of {batch_size}...")

    # Process in batches
    all_extractions = []  # Will match 1:1 with extraction_tasks
    successful_extractions = 0

    for i in tqdm(range(0, len(extraction_tasks), batch_size), desc="VLLM Batch Processing"):
        batch_tasks = extraction_tasks[i:i+batch_size]
        batch_prompts = [task['prompt'] for task in batch_tasks]
        
        # Generate batch with VLLM
        outputs = extractor.model.generate(batch_prompts, extractor.sampling_params)
        
        # Extract responses
        batch_results = []
        correct_extractions = 0
        results = []

        for i, output in enumerate(outputs):
            if output.outputs and len(output.outputs) > 0:
                response = output.outputs[0].text.strip()
                response = response.split('\n')[0].strip()
                response = response.replace('"', '').replace("'", "")
                is_correct = False
                if response and text.find(response) != -1:
                    is_correct = True
                    correct_extractions += 1

                results.append({
                    "text": task['i']['text'],
                    "target_equation": task['i']['tar_eq'],
                    "llm_extraction": response,
                    "is_correct": is_correct
                })
                batch_results.append(output)
        
        all_extractions.extend(batch_results)
        successful_extractions += correct_extractions
        # Print progress every few batches
        if (i // batch_size + 1) % 10 == 0:
            current_batch = i // batch_size + 1
            total_batches = (len(extraction_tasks) + batch_size - 1) // batch_size
            success_rate = (correct_extractions / len(all_extractions)) * 100 if all_extractions else 0
            print(f"  Batch {current_batch}/{total_batches} - Success rate so far: {success_rate:.1f}% ({correct_extractions}/{len(all_extractions)})")
    
    
    # 4. Save the results and print accuracy
    with open("results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Processing complete. Results saved to 'results.json'.")
    print(f"\n✅ Batch processing complete!")
    print(f"📊 Overall extraction success: {successful_extractions}/{len(all_extractions)} ({(successful_extractions/len(all_extractions)*100):.1f}%)")
    
    # Group extractions by sample index
    print("\n🔄 Grouping extractions by sample...")
    sample_extractions = {}  # sample_idx -> list of (eq_idx, extraction)
    
    for task_idx, extraction in enumerate(all_extractions):
        task = extraction_tasks[task_idx]
        sample_idx = task['sample_idx']
        eq_idx = task['eq_idx']
        
        if sample_idx not in sample_extractions:
            sample_extractions[sample_idx] = {}
        
        sample_extractions[sample_idx][eq_idx] = extraction
    
    # Now reconstruct the dataset with proper sample-equation mapping
    print("\n🔄 Reconstructing dataset with proper mapping...")
    successful_samples = 0
    failed_samples = []  # Track failed samples for analysis
    
    for i, sample in enumerate(tqdm(original_data, desc="Reconstructing dataset")):
        try:
            text = sample['text']
            target_equations = sample['tar_eq']
            
            new_start_indices = []
            new_end_indices = []
            success = True
            attempted_extractions = []
            failure_reason = None
            
            # Check if we have extractions for this sample
            if i not in sample_extractions:
                failure_reason = f"No extractions found for sample {i}"
                success = False
            else:
                for j, tar_eq in enumerate(target_equations):
                    if j in sample_extractions[i]:
                        math_expr = sample_extractions[i][j]
                        attempted_extractions.append(math_expr)
                        
                        if math_expr:
                            # Find token indices for the extracted expression
                            start_idx, end_idx = extractor.find_expression_in_tokens(text, math_expr, target_tokenizer)
                            
                            if start_idx is not None and end_idx is not None:
                                new_start_indices.append(start_idx)
                                new_end_indices.append(end_idx)
                            else:
                                print(f"  Sample {i}: Failed to map '{math_expr}' to tokens")
                                failure_reason = f"Failed to map '{math_expr}' to tokens in equation {j}"
                                success = False
                                break
                        else:
                            failure_reason = f"LLM could not extract expression for equation {j}: {tar_eq}"
                            success = False
                            break
                    else:
                        failure_reason = f"Missing extraction for equation {j}"
                        success = False
                        break
            
            if success:
                # Create new sample with corrected indices
                new_sample = {
                    'text': text,
                    'start_token_idx': new_start_indices,
                    'end_token_idx': new_end_indices,
                    'tar_eq': target_equations,
                    'tar_number': sample['tar_number']
                }
                retokenized_data.append(new_sample)
                successful_samples += 1
                
                # Show progress every 100 samples
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(original_data)} samples - Success: {successful_samples}/{i+1} ({(successful_samples/(i+1)*100):.1f}%)")
                    if attempted_extractions:
                        print(f"    Last sample extractions: {attempted_extractions}")
            else:
                skipped_samples += 1
                # Save failed sample for analysis
                failed_sample = {
                    'original_index': i,
                    'text': text,
                    'tar_eq': target_equations,
                    'tar_number': sample['tar_number'],
                    'attempted_extractions': attempted_extractions,
                    'failure_reason': failure_reason
                }
                failed_samples.append(failed_sample)
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            skipped_samples += 1
            # Save failed sample with error info
            failed_sample = {
                'original_index': i,
                'text': sample.get('text', ''),
                'tar_eq': sample.get('tar_eq', []),
                'tar_number': sample.get('tar_number', []),
                'attempted_extractions': [],
                'failure_reason': f"Exception: {str(e)}"
            }
            failed_samples.append(failed_sample)
    
    print(f"Retokenized dataset size: {len(retokenized_data)}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Success rate: {len(retokenized_data) / len(original_data) * 100:.1f}%")
    
    # Save retokenized dataset
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(retokenized_data, f, indent=2)
    
    print(f"Saved retokenized dataset to {output_file}")
    
    # Save failed samples for manual analysis
    if failed_samples:
        failed_file = output_file.replace('.json', '_failed.json')
        with open(failed_file, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        
        print(f"\n⚠️  Failed samples saved to: {failed_file}")
        print(f"📊 Failed samples breakdown:")
        
        # Analyze failure reasons
        failure_reasons = {}
        for failed in failed_samples:
            reason = failed['failure_reason']
            if reason in failure_reasons:
                failure_reasons[reason] += 1
            else:
                failure_reasons[reason] = 1
        
        for reason, count in failure_reasons.items():
            print(f"  - {reason}: {count} samples")
        
        # Show a few example failed cases
        print(f"\n🔍 Sample failed cases (first 3):")
        for i, failed in enumerate(failed_samples[:3]):
            print(f"\n  Failed Sample {failed['original_index']}:")
            print(f"    Text: {failed['text'][:100]}...")
            print(f"    Target equations: {failed['tar_eq']}")
            print(f"    Attempted extractions: {failed['attempted_extractions']}")
            print(f"    Failure reason: {failed['failure_reason']}")
    else:
        print("\n🎉 No failed samples! Perfect extraction!")
    
    return len(retokenized_data), len(failed_samples)

if __name__ == "__main__":
    import sys

    # Retokenize training data
    print("\n📚 Retokenizing training data...")
    llm_retokenize_dataset(
        input_file='data/gsm8k-xl/train.json',
        output_file='data/gsm8k-xl/train_qwen.json'
    )
    print("\n🎉 LLM-based retokenization completed!")
    print("You can now use the retokenized datasets for training with Qwen")
