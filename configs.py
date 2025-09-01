from dataclasses import dataclass
from typing import List

# Categories used for the examples
CATEGORIES = [
    "animal.n.01",
    "food.n.01",
    "vehicle.n.01",
    "tool.n.01",
    "clothing.n.01",
    "body_part.n.01",
    "location.n.01",
    "activity.n.01",
    "science.n.01",
    "sport.n.01",
    "building.n.01",
    "emotion.n.01",
    "communication.n.01",
    "event.n.01"
]

@dataclass
class DifficultyConfiguration:
    name_id: str = None
    num_targets: int = 3
    num_target_categories: int = 2
    num_intersected_categories: int = 2
    num_distractor_categories: int = 6
    categories: List[str] = None
    word_depth: int = 2
    rows: int = 4
    cols: int = 4
    seed: int = None

    def __post_init__(self):
        """Validate the difficulty configuration"""
        if self.categories is None:
            self.categories = CATEGORIES.copy()
        if self.name_id is None:
            self.name_id = self.generate_name_id()

        self._validate()

    def _validate(self):
        """Validate that the configuration makes sense"""
        # Basic positive integer checks
        if self.num_targets < 1:
            raise ValueError("num_targets must be at least 1.")
        if self.num_target_categories < 0:
            raise ValueError("num_target_categories must be non-negative.")
        if self.num_intersected_categories < 0:
            raise ValueError("num_intersected_categories must be non-negative.")
        if self.num_distractor_categories < 0:
            raise ValueError("num_distractor_categories must be non-negative.")
        if self.word_depth < 1:
            raise ValueError("word_depth must be at least 1.")
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be at least 1.")

        # Board size validation
        board_size = self.rows * self.cols
        if self.num_targets >= board_size:
            raise ValueError(f"num_targets ({self.num_targets}) must be less than board size ({board_size}).")

        # Category validation - unique categories needed
        # Intersected categories are counted once but used by both targets and distractors
        unique_categories_needed = self.num_target_categories + self.num_distractor_categories - self.num_intersected_categories
        max_available_categories = len(self.categories)

        if self.num_intersected_categories > self.num_target_categories:
            raise ValueError(f"num_intersected_categories must be <= num_target_categories.")
        if self.num_intersected_categories > self.num_distractor_categories:
            raise ValueError(f"num_intersected_categories must be <= num_distractor_categories.")
        if unique_categories_needed > max_available_categories:
            raise ValueError(f"Unique categories needed ({unique_categories_needed}) exceeds available categories ({max_available_categories}).")

    def generate_name_id(self):
        return f"diff_tgts={self.num_targets}_tcats={self.num_target_categories}_icats={self.num_intersected_categories}_dcats={self.num_distractor_categories}_depth={self.word_depth}_{self.rows}x{self.cols}_seed={self.seed}"

# ---------------------------------
# Configurations for the examples:
# ---------------------------------

EASY_CONFIG = DifficultyConfiguration(
    name_id='easy',
    num_targets=3,
    num_target_categories=1,
    num_intersected_categories=0,
    num_distractor_categories=4,
    word_depth=2,
    rows=4, cols=4
)

MEDIUM_CONFIG = DifficultyConfiguration(
    name_id='medium',
    num_targets=3,
    num_target_categories=3,
    num_intersected_categories=1,
    num_distractor_categories=3,
    word_depth=3,
    rows=4, cols=4
)

HARD_CONFIG = DifficultyConfiguration(
    name_id='hard',
    num_targets=4,
    num_target_categories=4,
    num_intersected_categories=3,
    num_distractor_categories=6,
    word_depth=4,
    rows=5, cols=5
)