import fs from "node:fs";
import { safetensors } from "@jax-js/loaders";
import { numpy as np, nn, lax } from "@jax-js/jax";
import { runGPT2, fromSafetensors } from "flax-js/models";

const CACHE_PATH = "/Users/caden/Programming/flax-js/.cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"

const buf = fs.readFileSync(CACHE_PATH);
const result = safetensors.parse(buf);

const model = fromSafetensors(result);
const x = np.array([[2215, 5335,  290, 1757, 1816,  284,  262, 3650,   11, 1757, 2921,  257,
    4144,  284]], { dtype: np.int32 });
const L = x.shape[1];
const positionIds = np.arange(L).astype(np.int32);
const output = runGPT2(model, x, positionIds, { numHeads: 12 });

const probs = nn.softmax(output.slice([], -1, []), -1);
const topk = lax.topK(probs, 10, -1);

const tokenProbs = topk[0].js();

console.log(tokenProbs);