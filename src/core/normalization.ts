import { numpy as np } from "@jax-js/jax";

function layerNorm(
    x: np.Array,
    bias: np.Array | null = null,
    scale: np.Array | null = null,
    eps: number = 1e-6,
): np.Array {
    let mean = np.mean(x, -1, { keepdims: true });
    let mean2 = np.mean(np.square(x), -1, { keepdims: true });
    let variance = mean2.sub(np.square(mean));
    let mul = np.rsqrt(variance.add(eps));

    if (scale !== null) {
        mul = np.matmul(mul, scale);
    }
    x = x.sub(mean).mul(mul);
    if (bias !== null) {
        x = x.add(bias);
    }
    return x;
}

export { layerNorm };
