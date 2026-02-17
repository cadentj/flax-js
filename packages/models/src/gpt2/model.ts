import { numpy as np, nn, jit } from "@jax-js/jax";
import {
    type Embed,
    runEmbed,
    type Linear,
    runLinear,
    type LayerNorm,
    runLayerNorm,
} from "flax-js";

type KVCache = {
    key: np.Array;
    value: np.Array;
};

export function emptyKVCache(dtype: np.DType): KVCache {
    return {
        key: np.zeros([0], { dtype }),
        value: np.zeros([0], { dtype }),
    };
}

type GPT2State = {
    kvCaches: KVCache[];
    kvCacheLen: number;
}

type GPT2Config = {
    numHeads: number;
};

export type GPT2 = {
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
    kvCache: KVCache,
    kvCacheLen: number,
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

    if (kvCacheLen > 0) {
        // Concat along the sequence dimension
        k = np.concatenate([kvCache.key, k], 1);
        v = np.concatenate([kvCache.value, v], 1);

        kvCache.key = k;
        kvCache.value = v;
        kvCacheLen += 1;
    }

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

const runGPT2Layer = jit(
    function runGPT2Layer(
        { ln1, attn, ln2, mlp }: GPT2Layer,
        x: np.Array,
        kvCache: KVCache,
        kvCacheLen: number,
        { numHeads }: GPT2Config,
    ): np.Array {
        let out = runGPT2Attention(attn, runLayerNorm(ln1, x.ref), kvCache, kvCacheLen, numHeads);
        x = x.add(out);
        out = runGPT2MLP(mlp, runLayerNorm(ln2, x.ref));
        x = x.add(out);
        return x;
    },
    { staticArgnums: [4] },
);

export const runGPT2 = jit(
    function runGPT2(
        { wte, wpe, h, lnF, lmHead }: GPT2,
        x: np.Array,
        positionIds: np.Array,
        state: GPT2State,
        { numHeads }: GPT2Config,
    ): np.Array {
        let out = runEmbed(wte, x);
        let positionEmbeds = runEmbed(wpe, positionIds);
        out = out.add(positionEmbeds);

        console.log(out.refCount);
        for (let layerIdx = 0; layerIdx < h.length; layerIdx++) {
            const layer = h[layerIdx];
            const kvCache = state.kvCaches[layerIdx];
            const kvCacheLen = state.kvCacheLen;
            out = runGPT2Layer(layer, out, kvCache, kvCacheLen, { numHeads });
        }
        out = runLayerNorm(lnF, out);
        out = runLinear(lmHead, out);
        return out;
    },
    { staticArgnums: [4] },
);

export function generateGPT2(
    model: GPT2,
    inputIds: np.Array,
    { numHeads }: GPT2Config,
    maxNewTokens: number,
): np.Array {
    // Build KV cache for each layer
    const kvCaches = []
    for (let i = 0; i < model.h.length; i++) {
        kvCaches.push(emptyKVCache(np.float32));
    }
    const state = {
        kvCaches,
        kvCacheLen: 0,
    };

    // Create position IDs
    const L = inputIds.shape[1];
    const positionIds = np.arange(L).astype(np.int32);

    // Run the model
    let out = runGPT2(model, inputIds, positionIds, state, { numHeads });

    for (let i = 0; i < maxNewTokens; i++) {
        out = runGPT2(model, out, positionIds, state, { numHeads });
    }
    return out;
}