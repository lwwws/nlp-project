# NLP Final Project: Codenames LLM Challenge

This project is submitted as the final project for the course **0970215 - Natural Language Processing**.

**Author:** Vsevolod Rusanov

---

## Table of Contents

* Project Idea
* Dataset Structure
* Data Generation Methodology
* Evaluation Methodology
* Running Instructions
* Project File Structure

---

## Project Idea

This project challenges an LLM's ability for semantic compression and preservation,
testing its reasoning and contextual understanding by creating a dataset inspired by the board game **Codenames**.
The game requires finding the best clue which abstracts the target words and avoids disambiguation from non-target words (distractors).

The intention behind the dataset is to evaluate how well an LLM can compress meaning into abstract clue,
and whether that abstraction can be preserved and faithfully unpacked back into intended concepts.

Codenames is known to be challenging even for humans. 
Success often requires a balance of logic, creativity, and unconventional thinking. 
For instance, human players sometimes use "meta-strategies" that go beyond pure semantics, giving clues related to word structure, phonetics, or even the position of words on the board.

While this project's dataset is constructed and evaluated mainly for semantic association, an LLM is not restricted from attempting these more advanced, unconventional strategies. Such behavior, even if not specifically rewarded by the current scoring metric, would demonstrate an even greater level of abstract reasoning and problem-solving.
The task is modeled as a two-step **clue-retrieval loop**, simplified to a one-sided round focusing on the Codemaster-Guesser dynamic:

### Codemaster Role

* **Input:** A game board with a subset of `k` target words.
* **Task:** Generate a single, one-word clue connecting the targets while avoiding distractors.

### Guesser Role

* **Input:** The clue, the value of `k`, and the full board.
* **Task:** Identify the intended target words from the board.

The evaluation focuses on the Guesser's accuracy as explained further.

---

## Dataset Structure

The dataset is stored at **`data/examples.jsonl`**, i.e. each line is a self-contained JSON object representing a game board and metadata.

### Example Structure:

```json
{
  "board": [["dog", "motor", "car", "wolf", "cat"],
            ["train", "bear", "..."]],
  "targets": ["dog", "cat"],
  "difficulty": "medium",
  "word_depth": 2,
  "target_categories": ["animal.n.01"],
  "distractor_categories": ["vehicle.n.01", "animal.n.01"],
  "intersected_categories": ["animal.n.01"]
}
```

**Fields:**

* `board (List[List[str]])`: 2D grid of all words.
* `targets (List[str])`: Intended `k` number of target words to connect.
* `difficulty (str)`: One of `easy`, `medium`, `hard`. Customizable via `DifficultyConfiguration`.
* `word_depth (int)`: Depth in WordNet hierarchy (higher = more specific/obscure).
* `target_categories (List[str])`: WordNet synsets for targets.
* `distractor_categories (List[str])`: WordNet synsets for distractors.
* `intersected_categories (List[str])`: WordNet synsets common to both groups (main challenge).

---

## Data Generation Methodology

1. **Semantic Sourcing from WordNet:** 
Words are sourced from WordNet. Boards are built around specific semantic categories, with `word_depth` selecting more specific hyponyms.
2. **Filtering with Brown Corpus:** 
Only common, recognizable words from the Brown Corpus are kept. Words must be single sequence, alphabetic, and of reasonable length. 
This was done to remove complex words and preserve the nature of original game.
3. **Controlled Board Construction:** 
Target and distractor words are selected from different (sometimes intersecting) categories. 
Additionally, board size, number of target words, word-depth are all control difficulty via `DifficultyConfiguration`.
---

## Evaluation Methodology

The evaluation focuses on the Guesser's accuracy, i.e., how well semantic information is preserved through the Codemaster -> Guesser communication loop.

**Primary Metric:** Precision@k - fraction of the Guesser's k guesses that are correct target words. 
This metric evaluates the success of the entire two-step process rather than the quality of the clue itself.

**Validation:** The script checks the LLM's response format (e.g., ensuring the right number of words are returned) before calculating the precision score.
We can easily modify the behaviour for invalid responses if necessary.

---

## Running Instructions

### 1. Setup

**Install Dependencies:**

```bash
pip install google-generativeai python-dotenv nltk
```

**API Key:** Create a `.env` file in the project root:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 2. Generate the Dataset

```bash
python generate_data.py
```

This creates **`data/examples.jsonl`**.

There are 3 custom pre-defined difficulties `EASY_CONFIG`, `MEDIUM_CONFIG`, `HARD_CONFIG`

* You can adjust how many examples per difficulty are generated by editing the `generation_plan` in `generate_data.py`.
* You can choose from which categories the words are taken by changing `CATEGORIES` in `configs.py`
* You can customize your own difficulty by creating custom `DifficultyConfiguration`

### 3. Run the Evaluation

```bash
python run_eval.py
```

This will:

1. Load `data/examples.jsonl`.
2. Iterate through each example.
3. Query the LLM for a clue (**Codemaster**).
4. Query the LLM again for guesses (**Guesser**).
5. Evaluate guesses and print results per round.
6. Print a final summary with the average score.

---

## Project File Structure

* **`data/examples.jsonl`** — Dataset file.
* **`generate_data.py`** — Dataset generation script.
* **`evaluation.py`** — Scoring function for model responses; validation logic can be customized.
* **`run_eval.py`** — Full evaluation pipeline.
* **`configs.py`** — Difficulty configurations.
* **`llm_api.py`** — API wrapper for Google GenAI.
* **`utils.py`** — Helper functions for data loading, board formatting, etc.
* **`README.md`** — Project documentation (this file).
