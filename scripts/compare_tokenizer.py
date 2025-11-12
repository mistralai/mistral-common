"""
This script compares the basic `.encode` tokenization
between Hugging Face and Mistral Common tokenizers
across multiple datasets.

It identifies and reports mismatches in tokenization results,
helping ensure consistency between the two tokenizers.
The comparison is limited to a user-specified number of characters
to avoid excessively long sequences.

The mismatch rate is calculated as the percentage of samples
where tokenization results differ at any point.
Results include the exact token where the mismatch starts,
as well as the three tokens immediately before and after,
totaling 7 tokens.

If no mismatch is detected, it only indicates that
the first `max_chars` characters are consistent—
not necessarily the entire sample.

Some datasets may show a high mismatch rate due to
frequent occurrences of the same problematic string.
We recommend reviewing the raw results for details.

Usage:
    python3 compare_tokenizer.py --hf_model <hf_model_name> --mc_model <mc_model_name>
        [--num_samples <num_samples>]
        [--max_chars <max_chars>]
        [--hf_token <hf_token>]
        [--save_results]

Arguments:
- `--hf_model`: Model name or path for the Hugging Face tokenizer.
- `--mc_model`: Model name or path for the Mistral-Common tokenizer.
  If not provided, defaults to `hf_model`.
- `--num_samples`: Maximum number of samples to test (default: 3000).
- `--max_chars`: Maximum number of characters to tokenize per sample (default: 10,000).
- `--hf_token`: Hugging Face token for private models (optional).
- `--save_results`: If provided, saves raw mismatch results to a JSON file
  (default: `tokenizer_mismatches_data.json`).

Examples:
    python3 compare_tokenizer.py
        --hf_model mistralai/Mistral-Small-3.1-24b-Instruct-2503
        --mc_model mistralai/Mistral-Small-3.1-24b-Instruct-2503
        --save_results
    python3 compare_tokenizer.py
        --hf_model mistralai/Mistral-Nemo-Instruct-2407
        --mc_model mistralai/Mistral-Small-3.1-24b-Instruct-2503
        --num_samples 1000
        --max_chars 5000
"""

import argparse
import json
from collections import Counter
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def compare_tokenizers(
    hf_tokenizer: AutoTokenizer,
    mc_tokenizer: MistralTokenizer,
    content: str,
    max_chars: int = 10_000,
) -> dict[str, Any] | None:
    """Compare tokenization between Hugging Face and Mistral tokenizers."""
    content = content[:max_chars]  # Limit to max_chars characters to avoid long sequences
    hf_tokens = hf_tokenizer.encode(content)
    mc_tokens = mc_tokenizer.instruct_tokenizer.tokenizer.encode(content, bos=True, eos=False)
    if hf_tokens != mc_tokens:
        mismatch_id = next(
            (idx for idx, (a, b) in enumerate(zip(hf_tokens, mc_tokens)) if a != b),
            min(len(hf_tokens), len(mc_tokens)),
        )
        start = max(0, mismatch_id - 3)
        end = min(len(hf_tokens), mismatch_id + 4)
        hf_str = [hf_tokenizer.decode([t]) for t in hf_tokens[start:end]]
        mc_str = [mc_tokenizer.decode([t], special_token_policy=SpecialTokenPolicy.KEEP) for t in mc_tokens[start:end]]
        return {
            "string_content": "".join(mc_str),
            "mistral_common": mc_str,
            "hugging_face": hf_str,
        }
    return None


def process_dataset(
    dataset_name: str,
    split: str,
    hf_tokenizer: AutoTokenizer,
    mc_tokenizer: MistralTokenizer,
    num_samples: int,
    max_chars: int,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Process a dataset and collect tokenizer mismatches."""
    mismatches = []
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for i, example in tqdm(
        enumerate(dataset),
        desc=f"Processing {dataset_name} ({split})",
        total=num_samples,
    ):
        if i >= num_samples:
            break
        content = example["text"]
        mismatch = compare_tokenizers(hf_tokenizer, mc_tokenizer, content, max_chars)
        if mismatch:
            mismatch.update(kwargs)
            mismatches.append(mismatch)
    return mismatches


def test_tokenizer(
    hf_model: str,
    mc_model: str,
    num_samples: int = 3000,
    max_chars: int = 10_000,
    hf_token: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Test tokenizer consistency across multiple datasets.
    Output Example:
    === SUMMARY ===
    Total Mismatch Rate: 1.87% ( 56 / 3000 )
     - HuggingFaceFW/fineweb: 0.60% ( 6 / 1000 )
     - HuggingFaceFW/finepdfs: 1.70% ( 17 / 1000 )
     - manu/project_gutenberg: 3.30% ( 33 / 1000 )
    === MISMATCH REPORT: TOP 10 FREQUENT TOKENS ===
    Total mismatched unique tokens analyzed: 199
    Token -> Frequency (across all mismatches)
    ----------------------------------------
    "'" -> 73 times
    ' O' -> 44 times
    ' Jimmy' -> 32 times
    ' (' -> 22 times
    '/' -> 21 times
    'Re' -> 18 times
    'Produ' -> 18 times
    ' and' -> 16 times
    'gan' -> 16 times
    'Reg' -> 16 times
    """
    print("Loading HF Tokenizer.")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)
    print("Loading Mistral-Common Tokenizer.")
    mc_tokenizer = MistralTokenizer.from_hf_hub(mc_model, token=hf_token)
    n_per_set = num_samples // 3
    # Web Data - Mostly English
    print(f"Testing Tokenizers on HuggingFaceFW/fineweb with {n_per_set} samples.")
    web_mismatches = process_dataset(
        "HuggingFaceFW/fineweb",
        "train",
        hf_tokenizer,
        mc_tokenizer,
        n_per_set,
        max_chars,
    )
    web_mismatch_rate = (len(web_mismatches) / n_per_set) * 100
    print(f"HuggingFaceFW/fineweb: {web_mismatch_rate:.2f}% Sample Mismatch ({len(web_mismatches)}/{n_per_set})")
    # Web PDF Data - Mostly English
    print(f"Testing Tokenizers on HuggingFaceFW/finepdfs with {n_per_set} samples.")
    pdf_mismatches = process_dataset(
        "HuggingFaceFW/finepdfs",
        "train",
        hf_tokenizer,
        mc_tokenizer,
        n_per_set,
        max_chars,
    )
    pdf_mismatch_rate = (len(pdf_mismatches) / n_per_set) * 100
    print(f"HuggingFaceFW/finepdfs: {pdf_mismatch_rate:.2f}% Sample Mismatch ({len(pdf_mismatches)}/{n_per_set})")
    # Multilingual Creative Writing Data
    print(f"Testing Tokenizers on manu/project_gutenberg with {n_per_set} samples.")
    multi_mismatches = []
    dataset_info = load_dataset("manu/project_gutenberg", streaming=True)
    splits = [s for s in list(dataset_info.keys()) if s != "en"]
    n_per_split = n_per_set // len(splits)
    collected = 0
    for split in splits:
        if collected >= n_per_set:
            break
        split_mismatches = process_dataset(
            "manu/project_gutenberg",
            split,
            hf_tokenizer,
            mc_tokenizer,
            n_per_split,
            max_chars,
        )
        multi_mismatches.extend(split_mismatches)
        collected += n_per_split
    if collected < n_per_set:
        remaining = num_samples - collected
        en_mismatches = process_dataset(
            "manu/project_gutenberg",
            "en",
            hf_tokenizer,
            mc_tokenizer,
            remaining,
            max_chars,
        )
        multi_mismatches.extend(en_mismatches)
    multi_mismatch_rate = (len(multi_mismatches) / n_per_set) * 100
    print(f"manu/project_gutenberg: {multi_mismatch_rate:.2f}% Sample Mismatch ({len(multi_mismatches)}/{n_per_set})")
    # Total Mismatch
    print("\n=== SUMMARY ===")
    rate = (len(web_mismatches) + len(pdf_mismatches) + len(multi_mismatches)) / num_samples * 100
    print(
        f"Total Mismatch Rate: {rate:.2f}% "
        f"( {len(web_mismatches) + len(pdf_mismatches) + len(multi_mismatches)} / {num_samples} )"
    )
    print(
        f" - HuggingFaceFW/fineweb: {len(web_mismatches) / n_per_set * 100:.2f}% "
        f"( {len(web_mismatches)} / {n_per_set} )"
    )
    print(
        f" - HuggingFaceFW/finepdfs: {len(pdf_mismatches) / n_per_set * 100:.2f}% "
        f"( {len(pdf_mismatches)} / {n_per_set} )"
    )
    print(
        f" - manu/project_gutenberg: {len(multi_mismatches) / n_per_set * 100:.2f}% "
        f"( {len(multi_mismatches)} / {n_per_set} )"
    )
    return {
        "web_model_mismatches": web_mismatches,
        "pdf_model_mismatches": pdf_mismatches,
        "multi_model_mismatches": multi_mismatches,
    }


def generate_mismatch_report(mismatch_results: dict[str, list[dict[str, Any]]]) -> None:
    """Generate a report of the most frequent tokens in mismatched samples."""
    all_tokens = []
    # Flatten all mismatches from all datasets
    for dataset_key in mismatch_results:
        for mismatch in mismatch_results[dataset_key]:
            hf_tokens = mismatch["hugging_face"]
            mc_tokens = mismatch["mistral_common"]
            all_tokens.extend(hf_tokens)
            all_tokens.extend(mc_tokens)
    # Count and rank tokens
    token_counter = Counter(all_tokens)
    most_common = token_counter.most_common(10)  # Top 10
    print("\n=== MISMATCH REPORT: TOP 10 FREQUENT TOKENS ===")
    print(f"Total mismatched unique tokens analyzed: {len(token_counter)}")
    print("Token -> Frequency (across all mismatches)")
    print("-" * 40)
    for token, count in most_common:
        print(f"{token!r} -> {count} times")


def save_results(
    results: dict[str, list[dict[str, Any]]],
    filename: str = "tokenizer_mismatches_data.json",
) -> None:
    """Save the results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tokenizer consistency between Hugging Face and Mistral Common.")
    parser.add_argument("--hf_model", type=str, help="Model name or path for the HF tokenizer.")
    parser.add_argument(
        "--mc_model",
        type=str,
        default=None,
        help="Model name or path for the Mistral-Common tokenizer.",
    )
    parser.add_argument("--n", type=int, default=3000, help="Maximum number of samples to test")
    parser.add_argument(
        "--max_chars",
        type=int,
        default=10_000,
        help="Maximum number of characters to tokenize per sample",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for private models",
    )
    parser.add_argument("--save_results", action="store_true", help="Save results to a JSON file")
    args = parser.parse_args()
    results = test_tokenizer(
        args.hf_model,
        args.mc_model if args.mc_model else args.hf_model,
        args.n,
        args.max_chars,
        args.hf_token,
    )
    generate_mismatch_report(results)
    if args.save_results:
        save_results(results)
