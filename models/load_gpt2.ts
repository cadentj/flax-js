import fs from "node:fs";
import { safetensors } from "@jax-js/loaders";
import { numpy as np, nn, lax } from "@jax-js/jax";

const buf = fs.readFileSync(".cache/model.safetensors");
const result = safetensors.parse(buf);

import { runGPT2 } from "flax-js/models/gpt2/model";
import { fromSafetensors } from "flax-js/models/gpt2/load";

const model = fromSafetensors(result);
const x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], { dtype: np.int32 });
const output = runGPT2(model, x, { numHeads: 12 });
