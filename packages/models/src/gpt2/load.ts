import { numpy as np } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";
import type { GPT2 } from "./model";

const weightMapper = new WeightMapper({
  substring: {
    ".ln_1.": ".ln1.",
    ".ln_2.": ".ln2.",
    ".attn.c_proj.": ".attn.o_proj.",
  },
  autoCamelCase: true,
});

export function fromSafetensors(file: safetensors.File): GPT2 {
  const hydrated: Record<string, np.Array> = {};

  for (const [key, tensor] of Object.entries(file.tensors)) {
    // Skip causal attention mask (not a trainable weight)
    if (/^h\.\d+\.attn\.bias$/.test(key)) continue;

    if (tensor.dtype !== "F32") {
      throw new Error(`Unexpected dtype ${tensor.dtype} for weight ${key}`);
    }

    const arr = np.array(tensor.data as Float32Array<ArrayBuffer>, {
      dtype: np.float32,
      shape: tensor.shape,
    });

    const mappedKey = weightMapper.mapKey(key);

    // Split combined QKV projection into separate Q, K, V
    const cAttnMatch = mappedKey.match(/^(.+\.attn)\.cAttn\.(weight|bias)$/);
    if (cAttnMatch) {
      const [, prefix, field] = cAttnMatch;
      // Conv1D weight is [D, 3D] -> transpose to [3D, D] then split into 3x [D, D]
      // Bias is [3D] -> split into 3x [D]
      const toSplit = field === "weight" ? arr.transpose() : arr;
      const [q, k, v] = np.split(toSplit, 3, 0);
      hydrated[`${prefix}.qProj.${field}`] = q;
      hydrated[`${prefix}.kProj.${field}`] = k;
      hydrated[`${prefix}.vProj.${field}`] = v;
      continue;
    }

    // Transpose Conv1D weights from [in, out] to [out, in] for standard Linear
    const isConv1DWeight =
      /\.oProj\.weight$/.test(mappedKey) ||
      /\.mlp\.cFc\.weight$/.test(mappedKey) ||
      /\.mlp\.cProj\.weight$/.test(mappedKey);

    hydrated[mappedKey] = isConv1DWeight ? arr.transpose() : arr;
  }

  // Weight tying: lmHead shares weights with wte embedding
  hydrated["lmHead.weight"] = hydrated["wte.weight"].ref;

  return safetensors.toNested(hydrated) as GPT2;
}
