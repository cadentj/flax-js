import fs from "node:fs";
import { safetensors } from "@jax-js/loaders";
import { numpy as np, nn, lax } from "@jax-js/jax";
import { generateGPT2, fromSafetensors, loadTokenizer } from "flax-js/models";

const CACHE_PATH = "/Users/caden/Programming/flax-js/.cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"

const startTime = performance.now();

const buf = fs.readFileSync(CACHE_PATH);
const result = safetensors.parse(buf);

const model = fromSafetensors(result);
const x = np.array([[2215, 5335,  290, 1757, 1816,  284,  262, 3650,   11, 1757, 2921,  257,
    4144,  284]], { dtype: np.int32 });

const output = generateGPT2(model, x, { numHeads: 12 }, 100);

const endTime = performance.now();
const duration = endTime - startTime;
console.log(`Time taken: ${duration} milliseconds`);

console.log(output.js());

// // TOKENIZER LOGIC

// import { tokenizers } from "@jax-js/loaders";

// const TOK_PATH = "/Users/caden/Programming/flax-js/.cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json"
// const TOK_CONFIG_PATH = "/Users/caden/Programming/flax-js/.cache/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer_config.json"

// const tokenizerData = JSON.parse(fs.readFileSync(TOK_PATH, 'utf-8'));
// const tokenizerConfig = JSON.parse(fs.readFileSync(TOK_CONFIG_PATH, 'utf-8'));

// const tok = loadTokenizer(tokenizerData, tokenizerConfig);

// tok.decode(output.js());

// console.log(tok.decode(output.js()));