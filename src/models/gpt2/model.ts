import { numpy as np, nn, jit } from "@jax-js/jax";
import {
    Embed,
    runEmbed,
    Linear,
    runLinear,
    LayerNorm,
    runLayerNorm,
} from "../../nn";

type GPT2Config = {
    numHeads: number;
};

type GPT2 = {
    wte: Embed;
    wpe: Embed;
    h: GPT2Layer[];
    lnF: LayerNorm;
    lmHead: Linear;
};

type GPT2Attention = {
    qProj: Linear;
    kProj: Linear;
    vProj: Linear;
    oProj: Linear;
};

const runGPT2Attention = jit(function runGPT2Attention(
    { qProj, kProj, vProj, oProj }: GPT2Attention,
    x: np.Array,
    numHeads: number,
): np.Array {
    const [B, L, D] = x.shape;
    const headDim = D / numHeads;

    // Project to q, k, v: [B, L, D]
    let q = runLinear(qProj, x);
    let k = runLinear(kProj, x);
    let v = runLinear(vProj, x);

    // Reshape to [B, L, H, K] for multi-head attention
    q = q.reshape([B, L, numHeads, headDim]);
    k = k.reshape([B, L, numHeads, headDim]);
    v = v.reshape([B, L, numHeads, headDim]);

    // Scaled dot-product attention with causal mask: [B, L, H, K]
    let out = nn.dotProductAttention(q, k, v, { isCausal: true });

    // Reshape back to [B, L, D] and apply output projection
    out = out.reshape([B, L, D]);
    out = runLinear(oProj, out);

    return out;
});

type GPT2Layer = {
    ln1: LayerNorm;
    attn: GPT2Attention;
    ln2: LayerNorm;
    mlp: Linear;
};

const runGPT2Layer = jit(function runGPT2Layer(
    { ln1, attn, ln2, mlp }: GPT2Layer,
    x: np.Array,
    { numHeads }: GPT2Config,
): np.Array {
    let out = runLayerNorm(ln1, x);
    out = runGPT2Attention(attn, out, numHeads);
    out = runLayerNorm(ln2, out);
    out = runLinear(mlp, out);
    return out;
});

const runGPT2 = jit(function runGPT2(
    { wte, wpe, h, lnF, lmHead }: GPT2,
    x: np.Array,
    { numHeads }: GPT2Config,
): np.Array {
    let out = runEmbed(wte, x);
    out = runEmbed(wpe, out);
    for (const layer of h) {
        out = runGPT2Layer(layer, out, { numHeads });
    }
    out = runLayerNorm(lnF, out);
    out = runLinear(lmHead, out);
    return out;
});

export { runGPT2, GPT2 }