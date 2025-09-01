import json

from nltk.corpus import wordnet as wn

def get_words_from_specific_depth(category_synset_name, common_words_filter, target_depth=1):
    """
    Gets words from a specific depth, BUT ONLY if they are in the common_words_filter.
    """
    words = set()
    start_synset = wn.synset(category_synset_name)
    current_level_synsets = [start_synset]

    for i in range(target_depth):
        next_level_synsets = []
        for synset in current_level_synsets:
            next_level_synsets.extend(synset.hyponyms())
        current_level_synsets = next_level_synsets
        if not current_level_synsets:
            return []

    for synset in current_level_synsets:
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            # Only add the word if it's a common word, and it is a single word.
            if word in common_words_filter and '_' not in word:
                words.add(word)

    return list(words)


def format_board(board_grid: list[list[str]]) -> str:
    """
    Formats a board grid into a single, nicely padded string.
    """
    # Create a list to hold each formatted row string
    formatted_rows = []

    # Process each row and format it
    for row in board_grid:
        # The f-string formatting pads each word to 14 characters
        formatted_row = " | ".join(f"{word:<14}" for word in row)
        formatted_rows.append(formatted_row)

    # Join all the row strings together with newlines
    return "\n".join(formatted_rows)

def load_dataset(file_path='data/examples.jsonl'):
    """Loads a dataset from a .jsonl file into a list of dictionaries."""
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # json.loads() parses a single JSON string (one line)
            data_point = json.loads(line)
            dataset.append(data_point)
    return dataset
