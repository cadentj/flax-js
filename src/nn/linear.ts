import { jit, numpy as np } from "@jax-js/jax";

export type Linear = {
    weight: np.Array;
    bias?: np.Array;
};

export const runLinear = jit(function runLinear(
    { weight, bias }: Linear,
    x: np.Array,
): np.Array {
    x = np.dot(x, weight.transpose());
    if (bias) x = x.add(bias);
    return x;
});

export type Embed = {
    embedding: np.Array;
};

export const runEmbed = jit(function runEmbed(
    { embedding }: Embed,
    x: np.Array,
): np.Array {
    return np.take(embedding, x, 0);
});
