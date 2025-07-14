#!/usr/bin/env python3
"""
Retokenization script for GSM8K-XL dataset
Converts LLaMA-tokenized indices to Qwen-tokenized indices
"""

import json
import re
import os
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm


class DatasetRetokenizer:
    """Retokenizes GSM8K-XL dataset for Qwen tokenizer"""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Loaded tokenizer: {tokenizer_name}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def extract_math_expression(self, tar_eq: str) -> str:
        """
        Extract the mathematical expression from target equation
        
        Args:
            tar_eq: "<divide>(48, 2)=24<eoe>" or "<add>(48, 24)=72<eoe>"
        
        Returns:
            math_expr: "48/2" or "48+24"
        """
        # Extract function and arguments - handle multiple arguments
        if tar_eq.startswith('<divide>'):
            # "<divide>(48, 2)=24<eoe>" -> "48/2"
            match = re.search(r'<divide>\(([^,]+),\s*([^)]+)\)', tar_eq)
            if match:
                arg1, arg2 = match.groups()
                return f"{arg1.strip()}/{arg2.strip()}"
        
        elif tar_eq.startswith('<add>'):
            # "<add>(48, 24)=72<eoe>" -> "48+24"
            # "<add>(+3, 3)=6<eoe>" -> "+ 3" (looking for pattern like "P + 3")
            match = re.search(r'<add>\(([^,]+),\s*([^)]+)\)', tar_eq)
            if match:
                arg1, arg2 = match.groups()
                arg1, arg2 = arg1.strip(), arg2.strip()
                
                # Handle case where first argument starts with + (like "+3")
                if arg1.startswith('+'):
                    # Look for pattern like "P + 3" or "something + 3"
                    return f"+ {arg1[1:]}"  # Remove the + and make it "+" + number
                else:
                    return f"{arg1}+{arg2}"
        
        elif tar_eq.startswith('<subtract>'):
            # Handle both 2-arg and multi-arg subtraction
            # "<subtract>(100, 25)=75<eoe>" -> "100-25"
            # "<subtract>(100, 50, 30, 15)=5<eoe>" -> "100 - 50 - 30 - 15"
            # "<subtract>(+5, 2)=3<eoe>" -> "- 2" (looking for pattern like "P - 2")
            match = re.search(r'<subtract>\((.*?)\)', tar_eq)
            if match:
                args = [arg.strip() for arg in match.group(1).split(',')]
                if len(args) >= 2:
                    # Check if first argument starts with + (like "+5")
                    if args[0].startswith('+'):
                        # Look for pattern like "P - number" or "something - number"
                        return f"- {args[1]}"  # Return "- number" to search for
                    else:
                        return f"{args[0]} - {' - '.join(args[1:])}"
        
        elif tar_eq.startswith('<multiply>'):
            # "<multiply>(12, 5)=60<eoe>" -> "12*5"
            match = re.search(r'<multiply>\(([^,]+),\s*([^)]+)\)', tar_eq)
            if match:
                arg1, arg2 = match.groups()
                return f"{arg1.strip()}*{arg2.strip()}"
        
        # If no pattern matched, try to extract from the equation format
        # Sometimes it might be in format like "48/2=24" directly
        eq_match = re.search(r'([0-9]+[+\-*/][0-9]+)', tar_eq)
        if eq_match:
            return eq_match.group(1)
        
        print(f"Warning: Could not extract math expression from: {tar_eq}")
        return None
    
    def find_expression_in_tokens(self, text: str, math_expr: str) -> Tuple[int, int]:
        """
        Find token indices where math_expr appears in tokenized text
        
        Args:
            text: Full text content
            math_expr: Mathematical expression like "48/2"
        
        Returns:
            (start_idx, end_idx): Token indices covering the expression
        """
        if not math_expr:
            return None, None
        
        # Try multiple variations of the expression
        variations = []
        
        # Original expression
        variations.append(math_expr)
        
        # With spaces around operators
        spaced = math_expr.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ')
        variations.append(spaced)
        
        # Without spaces around operators (compact version)
        compact = math_expr.replace(' + ', '+').replace(' - ', '-').replace(' * ', '*').replace(' / ', '/')
        variations.append(compact)
        
        # Handle decimal number variations (0.5 vs .5)
        if '0.' in math_expr:
            # Try replacing 0.X with .X
            decimal_var = math_expr.replace('0.', '.')
            variations.append(decimal_var)
            # Also add spaced and compact versions of decimal variant
            decimal_spaced = decimal_var.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ')
            variations.append(decimal_spaced)
            decimal_compact = decimal_var.replace(' + ', '+').replace(' - ', '-').replace(' * ', '*').replace(' / ', '/')
            variations.append(decimal_compact)
        
        # Handle special case for expressions starting with + or - (like "+ 3" or "- 2")
        if math_expr.startswith('+ ') or math_expr.startswith('- '):
            # For "+ 3", try patterns like "P + 3", "something + 3"
            # For "- 2", try patterns like "P - 2", "something - 2"
            operator = math_expr[0]  # + or -
            number = math_expr[2:]   # Get the number part
            
            # Use regex to find any character/word followed by operator number
            import re
            regex_pattern = r'\w+\s*' + re.escape(operator) + r'\s*' + re.escape(number)
            
            # Find all matches in text using regex
            matches = list(re.finditer(regex_pattern, text))
            if matches:
                # Use the first match
                match = matches[0]
                char_start = match.start()
                char_end = match.end()
                found_expr = match.group()
                
                # Skip the normal variation loop since we found it with regex
                variations = []  # Clear variations to skip the normal loop
            else:
                # Fallback to looking for just "operator number"
                variations.extend([f" {operator} {number}", f"{operator} {number}", f" {operator}{number}", f"{operator}{number}"])
        
        # Different multiplication symbols
        if '*' in math_expr:
            variations.append(math_expr.replace('*', ' x '))
            variations.append(math_expr.replace('*', ' × '))
            variations.append(math_expr.replace('*', 'x'))
            # Also try spaced versions
            variations.append(spaced.replace(' * ', ' x '))
            variations.append(spaced.replace(' * ', ' × '))
            # And compact versions
            variations.append(compact.replace('*', 'x'))
            variations.append(compact.replace('*', '×'))
            
            # Try reversed operand order for multiplication/division (commutative operations)
            # For "144*5" also try "5*144", "5 * 144", etc.
            if '*' in math_expr and not math_expr.startswith('+ ') and not math_expr.startswith('- '):
                parts = math_expr.split('*')
                if len(parts) == 2:
                    reversed_expr = f"{parts[1].strip()}*{parts[0].strip()}"
                    variations.append(reversed_expr)
                    variations.append(reversed_expr.replace('*', ' * '))
                    variations.append(reversed_expr.replace('*', ' x '))
                    variations.append(reversed_expr.replace('*', ' × '))
                    variations.append(reversed_expr.replace('*', 'x'))
                    variations.append(reversed_expr.replace('*', '×'))
            
            # For decimal variations with different multiplication symbols
            if '0.' in math_expr:
                decimal_var = math_expr.replace('0.', '.')
                variations.append(decimal_var.replace('*', ' x '))
                variations.append(decimal_var.replace('*', ' × '))
                variations.append(decimal_var.replace('*', 'x'))
                variations.append(decimal_var.replace('*', '×'))
                
                # Also try reversed order for decimal variants
                if '*' in decimal_var:
                    parts = decimal_var.split('*')
                    if len(parts) == 2:
                        reversed_decimal = f"{parts[1].strip()}*{parts[0].strip()}"
                        variations.append(reversed_decimal)
                        variations.append(reversed_decimal.replace('*', ' * '))
                        variations.append(reversed_decimal.replace('*', ' x '))
                        variations.append(reversed_decimal.replace('*', ' × '))
        
        # For addition, also try reversed order (commutative)
        if '+' in math_expr and not math_expr.startswith('+ '):
            parts = math_expr.split('+')
            if len(parts) == 2:
                reversed_expr = f"{parts[1].strip()}+{parts[0].strip()}"
                variations.append(reversed_expr)
                variations.append(reversed_expr.replace('+', ' + '))
        
        # Different division symbols
        if '/' in math_expr:
            variations.append(math_expr.replace('/', ' / '))
            variations.append(math_expr.replace('/', ' ÷ '))
            # And compact versions
            variations.append(compact.replace('/', '÷'))
            
            # For decimal variations with different division symbols
            if '0.' in math_expr:
                decimal_var = math_expr.replace('0.', '.')
                variations.append(decimal_var.replace('/', ' / '))
                variations.append(decimal_var.replace('/', ' ÷ '))
                variations.append(decimal_var.replace('/', '÷'))
        
        # Find character position using any variation (if not already found by regex)
        if not variations:  # Already found by regex above
            pass  # char_start and char_end already set
        else:
            char_start = -1
            char_end = -1
            found_expr = None
            
            for variation in variations:
                char_start = text.find(variation)
                if char_start != -1:
                    char_end = char_start + len(variation)
                    found_expr = variation
                    break
        
        if char_start == -1:
            print(f"Warning: Could not find '{math_expr}' in text")
            print(f"  Tried variations: {variations}")
            print(f"  Text snippet: '{text[:100]}'...")
            exit()
            return None, None
        
        # Tokenize text and map character positions to token positions
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Get character offsets for each token
        # This is a bit tricky since we need to map back from tokens to characters
        token_start_idx = None
        token_end_idx = None
        
        # Decode tokens one by one to find character mappings
        current_text = ""
        for i, token_id in enumerate(tokens):
            token_text = self.tokenizer.decode([token_id])
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
            print(f"Warning: Could not map '{found_expr}' to token indices")
            print(f"  Character span: {char_start}-{char_end}")
            print(f"  Text snippet: '{text[char_start:char_end]}'")
            return None, None
        
        return token_start_idx, token_end_idx
    
    def retokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retokenize a single sample
        
        Args:
            sample: Original sample with LLaMA indices
        
        Returns:
            retokenized_sample: Sample with Qwen indices
        """
        text = sample['text']
        target_equations = sample['tar_eq']
        
        # Extract mathematical expressions from target equations
        math_expressions = []
        for tar_eq in target_equations:
            math_expr = self.extract_math_expression(tar_eq)
            math_expressions.append(math_expr)
        
        # Find new token indices for each expression
        new_start_indices = []
        new_end_indices = []
        
        for math_expr in math_expressions:
            if math_expr:
                start_idx, end_idx = self.find_expression_in_tokens(text, math_expr)
                if start_idx is not None and end_idx is not None:
                    new_start_indices.append(start_idx)
                    new_end_indices.append(end_idx)
                else:
                    # Skip this sample if we can't find the expression
                    return None
            else:
                # Skip this sample if we can't extract the expression
                return None
        
        # Create new sample with corrected indices
        new_sample = {
            'text': text,
            'start_token_idx': new_start_indices,
            'end_token_idx': new_end_indices,
            'tar_eq': target_equations,
            'tar_number': sample['tar_number']
        }
        
        return new_sample
    
    def retokenize_dataset(self, input_file: str, output_file: str) -> None:
        """
        Retokenize entire dataset
        
        Args:
            input_file: Path to original dataset
            output_file: Path to save retokenized dataset
        """
        print(f"Loading dataset from {input_file}")
        with open(input_file, 'r') as f:
            original_data = json.load(f)
        
        print(f"Original dataset size: {len(original_data)}")
        
        retokenized_data = []
        skipped_samples = 0
        
        for i, sample in enumerate(tqdm(original_data, desc="Retokenizing")):
            try:
                new_sample = self.retokenize_sample(sample)
                if new_sample is not None:
                    retokenized_data.append(new_sample)
                else:
                    skipped_samples += 1
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                skipped_samples += 1
        
        print(f"Retokenized dataset size: {len(retokenized_data)}")
        print(f"Skipped samples: {skipped_samples}")
        print(f"Success rate: {len(retokenized_data) / len(original_data) * 100:.1f}%")
        
        # Save retokenized dataset
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(retokenized_data, f, indent=2)
        
        print(f"Saved retokenized dataset to {output_file}")
    
    def validate_retokenization(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that retokenized indices are correct
        
        Args:
            sample: Retokenized sample
        
        Returns:
            is_valid: True if indices are correct
        """
        text = sample['text']
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Check that all indices are within bounds
        for start_idx, end_idx in zip(sample['start_token_idx'], sample['end_token_idx']):
            if start_idx >= len(tokens) or end_idx > len(tokens):
                return False
            
            # Check that the token span makes sense
            if start_idx >= end_idx:
                return False
        
        return True


def main():
    """Main retokenization process"""
    print("🔄 Starting GSM8K-XL Dataset Retokenization for Qwen")
    print("=" * 60)
    
    # Initialize retokenizer
    retokenizer = DatasetRetokenizer()
    
    # Retokenize training data
    print("\n📚 Retokenizing training data...")
    retokenizer.retokenize_dataset(
        input_file='data/gsm8k-xl/train.json',
        output_file='data/gsm8k-xl/train_qwen.json'
    )
    
    # Retokenize test data
    print("\n🧪 Retokenizing test data...")
    retokenizer.retokenize_dataset(
        input_file='data/gsm8k-xl/test.json',
        output_file='data/gsm8k-xl/test_qwen.json'
    )
    
    # Validate a few samples
    print("\n✅ Validating retokenization...")
    with open('data/gsm8k-xl/train_qwen.json', 'r') as f:
        retokenized_data = json.load(f)
    
    validation_samples = retokenized_data[:10]  # Check first 10 samples
    valid_count = 0
    for i, sample in enumerate(validation_samples):
        is_valid = retokenizer.validate_retokenization(sample)
        if is_valid:
            valid_count += 1
        else:
            print(f"❌ Sample {i} failed validation")
    
    print(f"✅ Validation results: {valid_count}/{len(validation_samples)} samples valid")
    
    print("\n🎉 Retokenization completed!")
    print("You can now use 'data/gsm8k-xl/train_qwen.json' for training with Qwen")


if __name__ == "__main__":
    main()
