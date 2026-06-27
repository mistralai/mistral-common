"""
This script compares tokenization between Hugging Face and Mistral Common tokenizers
across multiple datasets and modes (e.g., text_encode, text_instruct).
It identifies and reports mismatches in tokenization results,
helping ensure consistency between the two tokenizers.
The comparison is limited to a user-specified number of tokens
to avoid excessively long sequences.
The mismatch rate is calculated as the percentage of samples
where tokenization results differ at any point.
Results include the exact token where the mismatch starts,
as well as the three tokens immediately before and after,
totaling 7 tokens.
If no mismatch is detected, it only indicates that
the first `max_tokens` tokens are consistent—
not necessarily the entire sample.
Some datasets may show a high mismatch rate due to
frequent occurrences of the same problematic string.
We recommend reviewing the raw results for details.

Usage:
    python3 compare_tokenizer.py --hf_model <hf_model_name> [--mc_model <mc_model_name>]
        [--config <config_path>]
        [--type <mode1> <mode2> ...]
        [--hf_token <hf_token>]
        [--save_results]

Arguments:
- `--hf_model`: Model name or path for the Hugging Face tokenizer (required).
- `--mc_model`: Model name or path for the Mistral-Common tokenizer.
  If not provided, defaults to `hf_model`.
- `--random_only`: If provided, only use random UTF-8 strings and no datasets for quick testing.
- `--config`: Path to JSON config file with mode-specific settings
  (default: "external/transformers/scripts/full_compare_tokenizer_config.json").
- `--type`: Type(s) of tokenization to perform (space-separated).
  Supported modes: text_encode, text_instruct, vision_encode, vision_instruct,
  tool_call_instruct, text_reasoning, vision_reasoning, tool_call_reasoning.
  Default: ["text_encode"].
- `--hf_token`: Hugging Face token for private models (optional).
- `--save_results`: If provided, saves raw mismatch results and report to JSON files
  (default: "tokenizer_mismatches_data.json" and "tokenizer_mismatches_report.json").

Examples:
    python3 compare_tokenizer.py
        --hf_model mistralai/Mistral-Small-3.1-24b-Instruct-2503
        --type text_encode text_instruct
        --save_results

    python3 compare_tokenizer.py
        --hf_model mistralai/Mistral-Nemo-Instruct-2407
        --mc_model mistralai/Mistral-Small-3.1-24b-Instruct-2503
        --config custom_config.json
        --type text_encode text_instruct
        --save_results

    python3 compare_tokenizer.py
        --hf_model mistralai/Mistral-Small-24b-Instruct-2501
        --type text_encode text_instruct
        --random_only
"""

import argparse
import json
import random
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def compare_tokenizers(
    hf_tokenizer: AutoTokenizer,
    mc_tokenizer: MistralTokenizer,
    content: str | dict | list,
    max_tokens: int = 2_048,
    mode: str = "text_encode",
) -> dict[str] | None:
    """Compare tokenization between Hugging Face and Mistral tokenizers."""
    if mode == "text_encode":
        assert isinstance(content, str), "Content must be a string for text_encode mode."
        content_str = content
        hf_tokens = hf_tokenizer.encode(content_str, add_special_tokens=False)[:max_tokens]
        mc_tokens = mc_tokenizer.instruct_tokenizer.tokenizer.encode(content_str, bos=False, eos=False)[:max_tokens]
    elif mode == "text_instruct":
        assert isinstance(content, list), "Content must be a list of messages for text_instruct mode."
        if content and content[-1]["role"] == "assistant":
            content = content[:-1]
        if content and content[0]["role"] != "system":
            content = [{"role": "system", "content": ""}] + content
        hf_tokens = hf_tokenizer.apply_chat_template(content, tokenize=True)[:max_tokens]
        request = ChatCompletionRequest(messages=content)
        mc_tokens = mc_tokenizer.encode_chat_completion(request).tokens[:max_tokens]
    else:
        raise NotImplementedError(f"Mode '{mode}' is not yet implemented.")
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
            "mode": mode,
        }
    return None


def process_dataset(
    dataset_name: str,
    split: str,
    column: str,
    hf_tokenizer: AutoTokenizer,
    mc_tokenizer: MistralTokenizer,
    num_samples: int,
    max_tokens: int,
    mode: str,
) -> list[dict[str]]:
    """Process a dataset and collect tokenizer mismatches."""
    mismatches = []
    if dataset_name not in ["random_utf8", "random_instruct_utf8"]:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        try:
            for i, example in tqdm(
                enumerate(dataset),
                desc=f"Processing {dataset_name} ({split}, mode: {mode})",
                total=num_samples,
            ):
                if i >= num_samples:
                    break
                content = example[column]
                mismatch = compare_tokenizers(hf_tokenizer, mc_tokenizer, content, max_tokens, mode)
                if mismatch:
                    mismatch.update(
                        {
                            "dataset_name": dataset_name,
                            "dataset_column": column,
                            "sample_index": i,
                        }
                    )
                    mismatches.append(mismatch)
        finally:
            if hasattr(dataset, "close"):
                print(f"Closing dataset {dataset_name}.")
                dataset.close()
    else:
        if dataset_name == "random_utf8":
            for i in tqdm(range(num_samples), desc=f"Processing random_utf8 (mode: {mode})"):
                content = "".join([chr(random.randint(0, 127)) for _ in range(max_tokens)])
                mismatch = compare_tokenizers(hf_tokenizer, mc_tokenizer, content, max_tokens, mode)
                if mismatch:
                    mismatch.update(
                        {
                            "dataset_name": dataset_name,
                            "dataset_column": column,
                            "sample_index": i,
                        }
                    )
                    mismatches.append(mismatch)
        elif dataset_name == "random_instruct_utf8":
            for i in tqdm(
                range(num_samples),
                desc=f"Processing random_instruct_utf8 (mode: {mode})",
            ):
                content = (
                    [
                        {
                            "role": "system",
                            "content": "".join([chr(random.randint(0, 127)) for _ in range(512)]),
                        }
                    ]
                    if random.random() < 0.5
                    else []
                )
                for i in range(random.randint(1, 10)):
                    if len(content) == 0:
                        content.append(
                            {
                                "role": "user",
                                "content": "".join([chr(random.randint(0, 127)) for _ in range(512)]),
                            }
                        )
                    else:
                        content.append(
                            {
                                "role": ("assistant" if content[-1]["role"] == "user" else "user"),
                                "content": "".join([chr(random.randint(0, 127)) for _ in range(512)]),
                            }
                        )
                mismatch = compare_tokenizers(hf_tokenizer, mc_tokenizer, content, max_tokens, mode)
                if mismatch:
                    mismatch.update(
                        {
                            "dataset_name": dataset_name,
                            "dataset_column": column,
                            "sample_index": i,
                        }
                    )
                    mismatches.append(mismatch)
    return mismatches


def test_tokenizer(
    hf_model: str,
    mc_model: str,
    random_only: bool,
    modes: list[str],
    hf_token: str | None = None,
    config: dict[str] = {},
) -> tuple[dict[str, list[dict[str]]], dict[str]]:
    """Test tokenizer consistency for each mode and dataset in the config.
    Returns: (mismatch_results, stats)
    """
    print("Loading HF Tokenizer.")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)
    print("Loading Mistral-Common Tokenizer.")
    mc_tokenizer = MistralTokenizer.from_hf_hub(mc_model, token=hf_token)
    all_mismatches = {}
    stats = {}

    for mode in modes:
        mode_config = config.get(mode, {})
        if not mode_config or "datasets" not in mode_config:
            print(f"Warning: No config found for mode '{mode}'. Skipping.")
            continue

        all_mismatches[mode] = []
        mode_total_samples = 0
        mode_total_mismatches = 0
        mode_dataset_stats = {}

        for dataset_config in mode_config["datasets"]:
            dataset_name = dataset_config["name"]
            if random_only and dataset_name not in [
                "random_utf8",
                "random_instruct_utf8",
            ]:
                continue
            split = dataset_config.get("split", "train")
            column = dataset_config.get("column", "text")
            num_samples = dataset_config.get("num_samples", 1000)
            max_tokens = dataset_config.get("max_tokens", 2048)

            print(
                f"\nTesting Tokenizers on {dataset_name} (column: {column}, mode: {mode}) with {num_samples} samples."
            )
            mismatches = process_dataset(
                dataset_name,
                split,
                column,
                hf_tokenizer,
                mc_tokenizer,
                num_samples,
                max_tokens,
                mode,
            )

            dataset_mismatches = len(mismatches)
            mismatch_rate = (dataset_mismatches / num_samples) * 100
            print(f"{dataset_name} ({mode}): {mismatch_rate:.2f}% Sample Mismatch ({dataset_mismatches}/{num_samples})")

            all_mismatches[mode].extend(mismatches)
            mode_total_samples += num_samples
            mode_total_mismatches += dataset_mismatches
            mode_dataset_stats[dataset_name] = {
                "total_samples": num_samples,
                "mismatches": dataset_mismatches,
                "mismatch_rate": mismatch_rate,
            }

        # Save mode-level stats
        mode_mismatch_rate = (mode_total_mismatches / mode_total_samples) * 100 if mode_total_samples > 0 else 0
        stats[mode] = {
            "total_samples": mode_total_samples,
            "total_mismatches": mode_total_mismatches,
            "mismatch_rate": mode_mismatch_rate,
            "datasets": mode_dataset_stats,
        }

    return all_mismatches, stats


def generate_mismatch_report(mismatch_results: dict[str, list[dict[str]]], stats: dict[str]) -> dict[str]:
    """Generate a detailed report using precomputed stats and return it as a dict."""
    report = {"summary_by_mode": {}, "top_tokens_by_mode": {}, "top_tokens_overall": {}}

    # Summary by mode
    for mode, mode_stat in stats.items():
        report["summary_by_mode"][mode] = {
            "total_samples": mode_stat["total_samples"],
            "total_mismatches": mode_stat["total_mismatches"],
            "mismatch_rate": mode_stat["mismatch_rate"],
            "datasets": mode_stat["datasets"],
        }

    # Top tokens by mode
    for mode in mismatch_results:
        mode_tokens = []
        for mismatch in mismatch_results[mode]:
            mode_tokens.extend(mismatch["hugging_face"])
            mode_tokens.extend(mismatch["mistral_common"])
        token_counter = Counter(mode_tokens)
        report["top_tokens_by_mode"][mode] = {
            "total_unique_tokens": len(token_counter),
            "top_5_tokens": token_counter.most_common(5),
        }

    # Top tokens overall
    all_tokens = []
    for mode in mismatch_results:
        for mismatch in mismatch_results[mode]:
            all_tokens.extend(mismatch["hugging_face"])
            all_tokens.extend(mismatch["mistral_common"])
    token_counter = Counter(all_tokens)
    report["top_tokens_overall"] = {
        "total_unique_tokens": len(token_counter),
        "top_10_tokens": token_counter.most_common(10),
    }

    # Print the report
    print("\n=== MISMATCH SUMMARY BY MODE ===")
    for mode, mode_stat in stats.items():
        print(f"\n--- Mode: {mode} ---")
        print(f"Total Samples: {mode_stat['total_samples']}")
        print(f"Total Mismatches: {mode_stat['total_mismatches']}")
        print(f"Mismatch Rate: {mode_stat['mismatch_rate']:.2f}%")
        print("Per-Dataset Breakdown:")
        for dataset, ds_stat in mode_stat["datasets"].items():
            print(
                f"  - {dataset}: {ds_stat['mismatch_rate']:.2f}% ({ds_stat['mismatches']}/{ds_stat['total_samples']})"
            )

    print("\n=== TOP 5 MISMATCHED TOKENS BY MODE ===")
    for mode in mismatch_results:
        mode_tokens = []
        for mismatch in mismatch_results[mode]:
            mode_tokens.extend(mismatch["hugging_face"])
            mode_tokens.extend(mismatch["mistral_common"])
        token_counter = Counter(mode_tokens)
        most_common = token_counter.most_common(5)
        print(f"\n--- Mode: {mode} ---")
        print(f"Total unique tokens: {len(token_counter)}")
        print("Token -> Frequency")
        print("-" * 30)
        for token, count in most_common:
            print(f"{token!r} -> {count} times")

    print("\n=== OVERALL TOP 10 MISMATCHED TOKENS (ALL MODES) ===")
    print(f"Total unique tokens analyzed: {len(token_counter)}")
    print("Token -> Frequency (across all mismatches)")
    print("-" * 40)
    for token, count in token_counter.most_common(10):
        print(f"{token!r} -> {count} times")

    return report


def save_results(
    results: dict[str, list[dict[str]]],
    report: dict[str],
    mismatches_filename: str = "tokenizer_mismatches_data.json",
    report_filename: str = "tokenizer_mismatches_report.json",
) -> None:
    """Save the results and report to JSON files."""
    with open(mismatches_filename, "w") as f:
        json.dump(results, f, indent=2)
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tokenizer consistency between Hugging Face and Mistral Common.")
    parser.add_argument(
        "--hf_model",
        type=str,
        required=True,
        help="Model name or path for the HF tokenizer.",
    )
    parser.add_argument(
        "--mc_model",
        type=str,
        default=None,
        help="Model name or path for the Mistral-Common tokenizer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="external/transformers/scripts/full_compare_tokenizer_config.json",
        help="Path to JSON config file with mode-specific settings.",
    )
    parser.add_argument(
        "--random_only",
        action="store_true",
        help="Only test random UTF-8 data (for quick testing).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face token for private models.",
    )
    parser.add_argument(
        "--type",
        type=str,
        nargs="+",
        default=["text_encode"],
        choices=[
            "text_encode",
            "vision_encode",
            "text_instruct",
            "vision_instruct",
            "tool_call_instruct",
            "text_reasoning",
            "vision_reasoning",
            "tool_call_reasoning",
        ],
        help="Type(s) of tokenization to perform (space-separated).",
    )
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON files.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    results, stats = test_tokenizer(
        args.hf_model,
        args.mc_model if args.mc_model else args.hf_model,
        args.random_only,
        args.type,
        args.hf_token,
        config,
    )

    report = generate_mismatch_report(results, stats)

    if args.save_results:
        save_results(results, report)
