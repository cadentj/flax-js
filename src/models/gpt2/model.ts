import { numpy as np, nn, jit } from "@jax-js/jax";
import {
    type Embed,
    runEmbed,
    type Linear,
    runLinear,
    type LayerNorm,
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

function runGPT2Attention(
    { qProj, kProj, vProj, oProj }: GPT2Attention,
    x: np.Array,
    numHeads: number,
): np.Array {
    const [B, L, D] = x.shape;
    const headDim = D / numHeads;

    // Project to q, k, v: [B, L, D]
    let q = runLinear(qProj, x.ref);
    let k = runLinear(kProj, x.ref);
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
}

type GPT2MLP = {
    cFc: Linear;
    cProj: Linear;
};

function runGPT2MLP({ cFc, cProj }: GPT2MLP, x: np.Array): np.Array {
    x = runLinear(cFc, x);
    x = nn.gelu(x);
    x = runLinear(cProj, x);
    return x;
}

type GPT2Layer = {
    ln1: LayerNorm;
    attn: GPT2Attention;
    ln2: LayerNorm;
    mlp: GPT2MLP;
};

const runGPT2Layer = jit(function runGPT2Layer(
    { ln1, attn, ln2, mlp }: GPT2Layer,
    x: np.Array,
    { numHeads }: GPT2Config,
): np.Array {
    let out = runGPT2Attention(attn, runLayerNorm(ln1, x.ref), numHeads);
    x = x.add(out);
    out = runGPT2MLP(mlp, runLayerNorm(ln2, x.ref));
    x = x.add(out);
    return x;
}, { staticArgnums: [2] });

const runGPT2 = jit(function runGPT2(
    { wte, wpe, h, lnF, lmHead }: GPT2,
    x: np.Array,
    { numHeads }: GPT2Config,
): np.Array {
    console.log("point 2", x.refCount);
    const L = x.shape[1];
    const positions = np.arange(L).astype(np.int32);
    let out = runEmbed(wte, x).add(runEmbed(wpe, positions));
    for (const layer of h) {
        console.log("point 3", out.refCount);
        out = runGPT2Layer(layer, out, { numHeads });
    }
    out = runLayerNorm(lnF, out);
    out = runLinear(lmHead, out);
    return out;
}, { staticArgnums: [2] });

export { runGPT2, GPT2 }