import { numpy as np, nn, jit, tree } from "@jax-js/jax";
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
};

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
    numHeads: number,
): [np.Array, KVCache] {
    const [B, L, D] = x.shape;
    const headDim = D / numHeads;

    let q = runLinear(qProj, x.ref);
    let k = runLinear(kProj, x.ref);
    let v = runLinear(vProj, x);

    q = q.reshape([B, L, numHeads, headDim]);
    k = k.reshape([B, L, numHeads, headDim]);
    v = v.reshape([B, L, numHeads, headDim]);

    const isPrefill = kvCache.key.size === 0;
    if (!isPrefill) {
        k = np.concatenate([kvCache.key, k], 1);
        v = np.concatenate([kvCache.value, v], 1);
    } else {
        tree.dispose(kvCache);
    }

    let out = nn.dotProductAttention(q, k.ref, v.ref, { isCausal: true });

    out = out.reshape([B, L, D]);
    out = runLinear(oProj, out);

    return [out, { key: k, value: v }];
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
        { numHeads }: GPT2Config,
    ): [np.Array, KVCache] {
        let out: np.Array;
        [out, kvCache] = runGPT2Attention(
            attn,
            runLayerNorm(ln1, x.ref),
            kvCache,
            numHeads,
        );
        x = x.add(out);
        out = runGPT2MLP(mlp, runLayerNorm(ln2, x.ref));
        x = x.add(out);
        return [x, kvCache];
    },
    { staticArgnums: [3] },
);

export function runGPT2(
    { wte, wpe, h, lnF, lmHead }: GPT2,
    x: np.Array,
    state: GPT2State,
    { numHeads }: GPT2Config,
): [np.Array, GPT2State] {
    const L = x.shape[1];
    let out = runEmbed(wte, x);
    const positionIds = np.arange(state.kvCacheLen, state.kvCacheLen + L).astype(np.int32);
    let positionEmbeds = runEmbed(wpe, positionIds);
    out = out.add(positionEmbeds);

    for (let layerIdx = 0; layerIdx < h.length; layerIdx++) {
        [out, state.kvCaches[layerIdx]] = runGPT2Layer(
            h[layerIdx], out, state.kvCaches[layerIdx], { numHeads },
        );
    }
    state.kvCacheLen += L;
    out = runLayerNorm(lnF, out);
    out = runLinear(lmHead, out);
    return [out, state];
}

export function generateGPT2(
    model: GPT2,
    inputIds: np.Array,
    { numHeads }: GPT2Config,
    maxNewTokens: number,
): np.Array {
    const kvCaches = [];
    for (let i = 0; i < model.h.length; i++) {
        kvCaches.push(emptyKVCache(np.float32));
    }
    let state: GPT2State = {
        kvCaches,
        kvCacheLen: 0,
    };

    // Prefill: run on full input sequence
    let out: np.Array;
    [out, state] = runGPT2(tree.ref(model), inputIds.ref, state, { numHeads });

    for (let i = 0; i < maxNewTokens; i++) {
        const lastToken = out.slice([], -1, []);
        const probs = nn.softmax(lastToken, -1);
        const pred = np.argmax(probs, -1).reshape([1, 1]);

        inputIds = np.concatenate([inputIds, pred.ref], 1);

        // Decode: only pass new token, KV cache has the rest
        [out, state] = runGPT2(tree.ref(model), pred, state, { numHeads });
    }
    return inputIds;
}
