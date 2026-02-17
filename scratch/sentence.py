# %%

import torch as t
import chz
from transformers import AutoTokenizer

import os

@chz.chz
class AccuracyBenchmark:
    repo_id: str

REPO_ID = "openai-community/gpt2"
TEXT = "When Mary and John went to the store, John gave a drink to"
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
inputs = tokenizer(TEXT, return_tensors="pt")

# %%

inputs.input_ids
