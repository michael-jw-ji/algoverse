import json
import random
from datasets import load_dataset

def generate_integer_distractors(correct_answer: str, num: int = 5):
    try:
        # Extract numeric part of the answer (remove commas, split on space or symbols)
        correct_int = int(float(correct_answer.replace(",", "").strip().split()[0]))
        distractors = set()
        while len(distractors) < num:
            offset = random.randint(-10, 10)
            distractor = correct_int + offset
            if distractor != correct_int:
                distractors.add(str(distractor))
        return list(distractors)
    except:
        # fallback if parsing fails
        return [f"{correct_answer} wrong {i}" for i in range(num)]

gsm8k = load_dataset("gsm8k", "main")
converted = {
    "name": "gsm8k_understanding",
    "description": "GSM8K converted into SuperGLUE-style format with integer distractors",
    "keywords": ["arithmetic reasoning", "grade school math"],
    "preferred_score": "multiple_choice_grade",
    "metrics": ["multiple_choice_grade"],
    "append_choices_to_input": False,
    "examples": []
}

for item in gsm8k["train"]:
    question = item["question"].strip()
    correct_answer = item["answer"].split("####")[-1].strip()
    distractors = generate_integer_distractors(correct_answer)
    target_scores = {ans: 0 for ans in distractors}
    target_scores[correct_answer] = 1

    converted["examples"].append({
        "input": question,
        "target_scores": target_scores
    })

# Save to JSON
with open("gsm8k_understanding.json", "w") as f:
    json.dump(converted, f, indent=2)
