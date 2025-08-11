from datasets import load_dataset
from typing import List, Dict


def extract_steps(answer: str) -> List[str]:
    return [step.strip() for step in answer.split("\n") if step.strip()]


def add_steps_to_examples(
    examples: List[Dict], answer_field: str = "answer"
) -> List[Dict]:
    for item in examples:
        if answer_field in item:
            item["steps"] = extract_steps(item[answer_field])
    return examples


def load_gsm8k_as_list(
    fields: List[str] = ["question", "answer"],
) -> List[Dict]:
    dataset = load_dataset("gsm8k", "main")

    all_examples = []
    for split_name in dataset:
        split_data = dataset[split_name]
        examples = [
            {k: example[k] for k in fields if k in example} for example in split_data
        ]
        examples = add_steps_to_examples(examples, answer_field="answer")
        all_examples.extend(examples)

    return all_examples


if __name__ == "__main__":
    data = load_gsm8k_as_list()
    for i, example in enumerate(data[:3]):
        print(f"Question {i + 1}: {example['question']}")
        print("Steps:")
        for j, step in enumerate(example["steps"], 1):
            print(f"  {j}. {step}")
        print()
