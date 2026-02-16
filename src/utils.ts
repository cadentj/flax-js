import { numpy as np } from "@jax-js/jax";

export function createCausalMask(seqLen: number): np.Array {
    // Lower-triangular matrix of 1s: [L, L]
    let mask = np.tril(np.ones([seqLen, seqLen]));
    // Convert to additive mask: 0 where attend, -Infinity where masked
    // (1 - mask) gives 1 for upper triangle, 0 for lower
    // Multiply by -Infinity to get the additive mask
    mask = np.ones([seqLen, seqLen]).sub(mask).mul(-Infinity);
    // Reshape to [1, 1, L, L] for broadcasting with [B, H, L, L]
    mask = np.reshape(mask, [1, 1, seqLen, seqLen]);
    return mask;
}
