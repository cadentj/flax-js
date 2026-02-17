from constants import OUTPUT_DIR

import torch as t
import chz
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import print_topk_table
import os

@chz.chz
class AccuracyBenchmark:
    repo_id: str

TEXT = "When Mary and John went to the store, John gave a drink to"

def execute(repo_id: str) -> None:

    print(os.getenv("HF_HOME"))
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    if tokenizer is None: 
        raise ValueError(f"Tokenizer not found for {repo_id}")

    inputs = tokenizer(TEXT, return_tensors="pt")
    output = model(**inputs)

    probs = t.softmax(output.logits[:, -1, :], -1)
    topk = t.topk(probs, 10, -1)

    token_strings = [tokenizer.decode(token) for token in topk.indices[0]]
    token_probs = topk.values[0]

    model_name = repo_id.split("/")[-1]
    output_path = os.path.join(OUTPUT_DIR, f"torch-{model_name}.txt")
    print_topk_table(token_strings, token_probs, file=output_path)


if __name__ == "__main__":
    args = chz.entrypoint(AccuracyBenchmark)
    execute(args.repo_id)
