// Implementation of LayerNorm from Eric Zhang's PocketTTS
// https://github.com/ekzhang/jax-js/blob/main/website/src/routes/tts/pocket-tts.ts

import { numpy as np, jit } from "@jax-js/jax";

export type LayerNorm = {
    // LayerNorm with `elementwise_affine`, i.e. has weight and bias
    weight: np.Array;
    bias: np.Array;
};

export const runLayerNorm = jit(
    function runLayerNorm(
        { weight, bias }: Partial<LayerNorm> = {},
        x: np.Array,
        eps: number = 1e-5,
    ) {
        const dtype = x.dtype;
        x = x.astype(np.float32); // LayerNorm in high precision to avoid numerics issues.
        const mean = x.ref.mean(-1, { keepdims: true });
        const var_ = np.var_(x.ref, -1, {
            mean: mean.ref,
            correction: 0,
            keepdims: true,
        });
        x = x.sub(mean).div(np.sqrt(var_.add(eps)));
        if (weight) {
            x = x.mul(weight).add(bias!);
        }
        return x.astype(dtype);
    },
    { staticArgnums: [2] },
);
