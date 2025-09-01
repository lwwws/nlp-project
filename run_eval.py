from statistics import mean
from typing import Dict, Any

from evaluation import validate_codemaster_response
from llm_api import call_llm_api
from utils import format_board, load_dataset
from evaluation import eval

def format_codemaster_prompt(datapoint) -> str:
    """Creates the prompt for the codemaster LLM call."""
    # Helper to format the board into a string
    board_str = format_board(datapoint['board'])
    targets = datapoint['targets']

    return \
f"""
You are playing a round of Codenames. You are a Codemaster. Your goal is to provide a single-word clue to help your teammate guess the target words.
The target words are: {', '.join(targets)}.
Do NOT use any of the words on the board as your clue.

Here is the game board:
---
{board_str}
---

Based on the target words {', '.join(targets)}, provide a single clue word.
Your response must contain ONLY the clue word inside a code block. For example: ```ClueWord```
"""

def format_guesser_prompt(datapoint, clue: str) -> str:
    """Creates the prompt for the guesser LLM call."""
    board_str = format_board(datapoint['board'])
    k = len(datapoint['targets'])

    return \
f"""
You are playing a round of Codenames. You are a Guesser. Your teammate has given you a clue and a number.
Your goal is to guess {k} words from the board that are related to the clue.

Here is the game board:
---
{board_str}
---

The clue is: "{clue}"
The number is: {k}

List exactly {k} words from the board that you think are the targets.
Your response must contain ONLY the list of guessed words inside a code block, separated by newlines.
For example, if k=3:
```
word1
word2
word3
```
"""

def run_single_datapoint(datapoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the full codemaster -> guesser -> evaluation pipeline for a single data point.
    Returns a dictionary with the outcome.
    """
    # 1. Codemaster Turn
    codemaster_prompt = format_codemaster_prompt(datapoint)
    codemaster_response = call_llm_api(codemaster_prompt)
    if codemaster_response is None:
        return {"status": "codemaster_api_failure", "score": 0.0}

    is_valid_clue, clue_or_error = validate_codemaster_response(codemaster_response)
    if not is_valid_clue:
        print(f"Codemaster response was INVALID: {clue_or_error}")
        return {"status": "codemaster_validation_failure", "score": 0.0}

    clue = clue_or_error
    print(f"Codemaster clue is VALID: '{clue}'")

    # 2. Guesser Turn
    guesser_prompt = format_guesser_prompt(datapoint, clue)
    guesser_response = call_llm_api(guesser_prompt)
    if guesser_response is None:
        return {"status": "guesser_api_failure", "score": 0.0}

    # 3. Evaluation
    score = eval(guesser_response, datapoint)
    print(f"Score for this example: {score:.2f}")
    return {"status": "success", "score": score}

def main():
    """
    Loads the dataset, evaluates each data point, and reports the final summary.
    """
    dataset = load_dataset(file_path='data/examples.jsonl')
    if not dataset:
        print("Dataset is empty. Please generate it first.")
        return

    results = []
    total_examples = len(dataset)

    for i, datapoint in enumerate(dataset):
        print(f"\n--- Processing Example {i + 1}/{total_examples} (Difficulty: {datapoint['difficulty']}) ---")
        result = run_single_datapoint(datapoint)
        results.append(result)

    # --- Final Report ---
    scores = [r['score'] for r in results if r['status'] == 'success']
    codemaster_failures = sum(1 for r in results if 'codemaster' in r['status'])
    guesser_failures = sum(1 for r in results if 'guesser' in r['status'])

    print("\n\n--- EVALUATION COMPLETE ---")
    if scores:
        print(f"Average Score (Precision@k) across {len(scores)} successful rounds: {mean(scores):.4f}")
    else:
        print("No rounds were successfully completed.")

    print(f"Total Examples Processed: {total_examples}")
    print(f"Codemaster Failures (API or Validation): {codemaster_failures}")
    print(f"Guesser API Failures: {guesser_failures}")
    print(f"Successfully Evaluated Rounds: {len(scores)}")

if __name__ == "__main__":
    main()