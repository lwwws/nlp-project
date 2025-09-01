import json
import re

def eval(guesser_response: str, datapoint: dict) -> float:
    """
        Evaluates the guesser's response for a single data point.
    """
    all_words = get_all_words(datapoint['board'])
    targets = datapoint['targets']
    k = len(targets)

    is_valid, guesses = validate_guesser_response(guesser_response, k, all_words)

    if is_valid:
        return precision(guesses, targets)
    else:
        return 0.0


def precision(guesses, targets) -> float:
    """
    Calculates the precision@k for a Codenames board.

    Args:
        guesses (list or set): The list of words guessed by the LLM.
        targets (list or set): The list of correct target words.

    Returns:
        float: The precision score.
    """
    # Ensure targets and guesses are sets for easy intersection
    target_set = set(targets)
    guess_set = set(guesses)
    k = len(target_set)

    # This should be equal to k if the LLM follows instructions
    num_guesses = len(guess_set)
    if num_guesses == 0:
        return 0.0

    correct_guesses = len(target_set.intersection(guess_set))
    return correct_guesses / k

def get_all_words(board: list[list[str]]) -> set[str]:
    """Flattens a 2D board grid"""
    return {word.lower() for row in board for word in row}

def validate_codemaster_response(response: str):
    """
    Validates the codemaster's response, expecting a single clue word inside a code block.
    """
    try:
        # Extract content from the code block first.
        match = re.search(r"```(.*?)```", response, re.DOTALL)

        if not match:
            return False, "Error: Could not find a code block (```...```) in the response."

        # The actual content to parse is the text captured by the regex
        content = match.group(1).strip()
        words = content.lower().split()

        if len(words) != 1:
            return False, f"Error: Clue must be a single word, but found {len(words)} in the code block."

        clue = words[0]

        if not clue.isalpha():
            return False, f"Error: Clue '{clue}' contains non-alphabetic characters."

        return True, clue
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"


def validate_guesser_response(response: str, k: int, all_words: set[str]):
    """
    Validates the guesser's response, expecting n words inside a code block.
    """
    try:
        # New: Extract content from the code block first.
        match = re.search(r"```(.*?)```", response, re.DOTALL)

        if not match:
            return False, "Error: Could not find a code block (```...```) in the response."

        content = match.group(1).strip()

        # Now parse the content from inside the block
        guesses = [word.lower() for word in re.findall(r'\b[a-zA-Z]+\b', content)]

        if len(guesses) != k:
            return False, f"Error: Expected {k} guesses, but found {len(guesses)} in the code block."

        invalid_words = [guess for guess in guesses if guess not in all_words]

        if invalid_words:
            return False, f"Error: The following guesses from the code block are not on the board: {', '.join(invalid_words)}."

        # If needed, we can allow for some invalid response here easily
        # > Trimming to first k guesses if > k
        #
        # > Only focusing on valid words and returning:
        # [guess for guess in guesses if guess in all_words]

        return True, guesses
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

