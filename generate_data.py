import json
import os
import random
from configs import DifficultyConfiguration, EASY_CONFIG, MEDIUM_CONFIG, HARD_CONFIG

import nltk

from utils import get_words_from_specific_depth, format_board

nltk.download('wordnet')
nltk.download('brown')

from nltk.corpus import brown

# We create this once when the script starts.
print("Loading common words filter...")

# We use a set for very fast lookups.
COMMON_WORDS = set(w.lower() for w in brown.words() if len(w) >= 3 and w.isalpha())
print(f"Filter created with {len(COMMON_WORDS)} common words.")

def generate_board(diff: DifficultyConfiguration):
    """
    Generates a Codenames board based on difficulty configuration.
    Uses intersected categories for both targets and distractors.
    """
    random.seed(diff.seed)
    board_size = diff.rows * diff.cols

    categories = diff.categories.copy()
    random.shuffle(categories)

    # Calculate how many unique categories we need
    # Intersected categories count once but are used by both targets and distractors
    unique_categories_needed = (diff.num_target_categories +
                               diff.num_distractor_categories -
                               diff.num_intersected_categories)

    # Select all unique categories we'll use
    selected_categories = categories[:unique_categories_needed]

    # Split categories into intersected and non-intersected
    intersected_cats = selected_categories[:diff.num_intersected_categories]
    remaining_cats = selected_categories[diff.num_intersected_categories:]

    # Calculate how many non-intersected categories each group gets
    target_only_cats_needed = diff.num_target_categories - diff.num_intersected_categories
    distractor_only_cats_needed = diff.num_distractor_categories - diff.num_intersected_categories

    # Assign remaining categories
    target_only_cats = remaining_cats[:target_only_cats_needed]
    distractor_only_cats = remaining_cats[target_only_cats_needed:target_only_cats_needed + distractor_only_cats_needed]

    # Build final category lists
    target_cats = intersected_cats + target_only_cats
    distractor_cats = intersected_cats + distractor_only_cats

    # Generate target words
    target_pool = []
    depth = diff.word_depth

    while len(target_pool) < diff.num_targets and depth >= 0:
        for cat in target_cats:
            words = get_words_from_specific_depth(cat, COMMON_WORDS, depth)
            target_pool.extend(words)

        target_pool = list(set(target_pool))
        depth -= 1

    if len(target_pool) < diff.num_targets:
        raise ValueError("Not enough target words available, even after depth fallback")

    random.shuffle(target_pool)
    target_words = target_pool[:diff.num_targets]

    # Generate distractor words
    depth = diff.word_depth
    num_distractors = board_size - len(target_words)
    distractor_pool = []

    while len(distractor_pool) < num_distractors and depth >= 0:
        for cat in distractor_cats:
            words = get_words_from_specific_depth(cat, COMMON_WORDS, depth)
            distractor_pool.extend(words)

        distractor_pool = list(set(distractor_pool))
        distractor_pool = [w for w in distractor_pool if w not in target_words]
        depth -= 1

    if len(distractor_pool) < num_distractors:
        raise ValueError("Not enough distractor words available, even after depth fallback")

    random.shuffle(distractor_pool)
    distractor_words = distractor_pool[:num_distractors]

    # Create the board
    all_words = target_words + distractor_words
    random.shuffle(all_words)
    grid = [all_words[i * diff.cols:(i + 1) * diff.cols] for i in range(diff.rows)]

    return {
        "board": grid,
        "targets": sorted(target_words),
        "target_categories": target_cats,
        "distractor_categories": distractor_cats,
        "intersected_categories": intersected_cats
    }

def generate_data():
    """
    Generates a dataset with a specified number of examples for each difficulty level
    and saves it to a .jsonl file.
    """
    # We can define different generation plans via a list of (config, num_examples) tuples
    # This makes it easy to control how many of each type we want.

    # To generate a custom difficulty, you can just change the plan
    # custom_config = DifficultyConfiguration(name_id='my_custom_test', ...)
    # generation_plan = [(custom_config, 10)]

    generation_plan = [
        (EASY_CONFIG, 111),
        (MEDIUM_CONFIG, 111),
        (HARD_CONFIG, 111)
    ]

    dataset = []
    total_to_generate = sum(num for _, num in generation_plan)
    global_seed_offset = 0
    print(f"--- Generating {total_to_generate} Total Examples ---")

    for config, num_examples in generation_plan:
        print(f"\nGenerating {num_examples} '{config.name_id}' examples...")

        total_generated = 0

        while total_generated < num_examples:
            success = False
            attempts = 0
            max_attempts = 10

            while not success and attempts < max_attempts:
                try:
                    # Use the global offset and add attempts for retries
                    config.seed = global_seed_offset + attempts
                    data_point = generate_board(config)
                    success = True
                except ValueError as e:
                    print(f"Attempt {attempts + 1} failed for seed {config.seed}: {e}")
                    attempts += 1

            if not success:
                raise RuntimeError(f"Could not generate a board for {config.name_id} after {max_attempts} attempts")

            # Increment the global offset only AFTER a successful generation
            global_seed_offset += attempts + 1
            total_generated += 1

            # Add metadata
            data_point['difficulty'] = config.name_id
            data_point['word_depth'] = config.word_depth
            dataset.append(data_point)

    # Save the dataset to the required file format
    if not os.path.exists('data'):
        os.makedirs('data')

    output_path = 'data/examples.jsonl'
    with open(output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"\nSuccessfully generated and saved {len(dataset)} examples to '{output_path}'.")

if __name__ == "__main__":
    generate_data()