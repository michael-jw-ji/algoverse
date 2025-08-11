import os
import json
import random
import string
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader

CHOICES = [f"({v})" for v in string.ascii_uppercase[:6]]
COT = """\
Q: Mary had 3 apples. She bought 2 more. How many apples does she have now?
Options:
(A) 4
(B) 5
(C) 6
(D) 3
(E) 2
(F) 7
A: Let's think step by step.
Mary had 3 apples. She bought 2 more, so now she has 3 + 2 = 5 apples. So the answer is (B).

Q: Tom had 10 candies and gave away 4 to his friend. How many does he have left?
Options:
(A) 6
(B) 4
(C) 5
(D) 7
(E) 8
(F) 10
A: Let's think step by step.
Tom had 10 candies and gave away 4, so he has 10 - 4 = 6 left. So the answer is (A).
"""

PROB_HEADER = "Q: "


class GSM8KDataset(BaseDataset):
    """Grade School Math (GSM8K)-style arithmetic dataset."""

    description = "Answer grade school math word problems."
    data_file = "gsm8k_understanding.json"

    def __init__(self, n=1000):
        super().__init__()
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    def get_questions(self):
        task = json.load(open(Path(__file__).parent / "data" / self.data_file))
        for ex in task["examples"]:
            single_input = ex["input"] + "\nOptions:"
            single_target = ""
            targets = list(ex["target_scores"].items())
            random.shuffle(targets)
            for i, (k, v) in enumerate(targets):
                k = k.replace (",", "")
                k = str(int(float(k)))  # ensure integer format if decimals present
                single_input += "\n" + f"{CHOICES[i]} {k}"
                if v == 1:
                    single_target = CHOICES[i]
            self._examples.append({"input": single_input, "target": single_target})
        random.shuffle(self._examples)
        self._examples = self._examples[: self.n]

    def format_questions(self, formatter: PromptFormatter):
        global PROB_HEADER, COT
        Qs, As = [PROB_HEADER + f"{v['input']}" for v in self._examples], [
            f"\nA: {v['target']}" for v in self._examples
        ]
        answer_starter = "\nA: "
        if formatter.name == "chain-of-thought":
            answer_starter = "\nA: Let's think step by step."
        self._clean_examples = [
            formatter.format(
                task_description=self.description,
                prompt=v + answer_starter,
                questions=Qs,
                answers=As,
                cot=COT,
            )
            for v in Qs
        ]
        self._labels = [v["target"] for v in self._examples]
        for i in range(len(self._clean_examples)):
            # get a corrupted version with a different answer
            diff = list(
                filter(
                    lambda j: self._labels[j] != self._labels[i],
                    range(len(self._examples)),
                )
            )
            self._corrupted_examples.append(self._clean_examples[random.choice(diff)])

    def to_dataloader(self, model, batch_size: int, collate_fn=None):
        collate_fn = partial(generic_collate, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return (
            self._clean_examples[idx],
            self._corrupted_examples[idx],
            self._labels[idx],
        )
