import { numpy as np } from "@jax-js/jax";
import { nn } from "@jax-js/jax";
import { Embed, Linear, LayerNorm, Module, Params } from "../nn";
import { createCausalMask } from "../utils";

/**
Dimension key:

B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
**/


class GPT2Config {
    sequenceLen: number;
    vocabSize: number;
    nLayer: number;
    nHead: number;
    nEmbd: number;

    constructor(
        sequenceLen: number = 1024,
        vocabSize: number = 50257,
        nLayer: number = 12,
        nHead: number = 12,
        nEmbd: number = 768,
    ) {
        this.sequenceLen = sequenceLen;
        this.vocabSize = vocabSize;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nEmbd = nEmbd;
    }
}

class GPT2Attention extends Module {
    nHead: number;
    nEmbd: number;
    headDim: number;
    qProj: Linear;
    kProj: Linear;
    vProj: Linear;
    oProj: Linear;

    constructor(config: GPT2Config) {
        super();

        this.nHead = config.nHead;
        this.nEmbd = config.nEmbd;
        this.headDim = this.nEmbd / this.nHead;
        this.qProj = new Linear(this.nEmbd, this.nEmbd);
        this.kProj = new Linear(this.nEmbd, this.nEmbd);
        this.vProj = new Linear(this.nEmbd, this.nEmbd);
        this.oProj = new Linear(this.nEmbd, this.nEmbd);
    }

    forward(params: Params, x: np.Array, mask: np.Array): np.Array {
        const shape = x.shape;
        const B = shape[0];
        const L = shape[1];
        const H = this.nHead;
        const K = this.headDim;

        // Project to q, k, v: [B, L, D]
        let q = this.qProj.forward(params.qProj as Params, x);
        let k = this.kProj.forward(params.kProj as Params, x);
        let v = this.vProj.forward(params.vProj as Params, x);

        // Reshape to [B, L, H, K] then transpose to [B, H, L, K]
        q = np.transpose(np.reshape(q, [B, L, H, K]), [0, 2, 1, 3]);
        k = np.transpose(np.reshape(k, [B, L, H, K]), [0, 2, 1, 3]);
        v = np.transpose(np.reshape(v, [B, L, H, K]), [0, 2, 1, 3]);

        // Scaled dot-product attention: [B, H, L, L]
        let scores = np.matmul(q, np.transpose(k, [0, 1, 3, 2]));
        scores = scores.mul(1 / Math.sqrt(K));
        scores = scores.add(mask);
        const weights = nn.softmax(scores, -1);
        let out = np.matmul(weights, v); // [B, H, L, K]

        // Reshape back to [B, L, D]
        out = np.transpose(out, [0, 2, 1, 3]);
        out = np.reshape(out, [B, L, this.nEmbd]);
        return this.oProj.forward(params.oProj as Params, out);
    }
}


class GPT2MLP extends Module {
    upProj: Linear;
    downProj: Linear;

    constructor(config: GPT2Config) {
        super();
        this.upProj = new Linear(config.nEmbd, config.nEmbd * 4);
        this.downProj = new Linear(config.nEmbd * 4, config.nEmbd);
    }

    forward(params: Params, x: np.Array): np.Array {
        let h = this.upProj.forward(params.upProj as Params, x);
        h = nn.gelu(h, { approximate: true });
        h = this.downProj.forward(params.downProj as Params, h);
        return h;
    }
}

class GPT2Block extends Module {
    self_attn: GPT2Attention;
    mlp: GPT2MLP;
    inputLayerNorm: LayerNorm;
    postAttentionLayerNorm: LayerNorm;

    constructor(config: GPT2Config) {
        super();
        this.inputLayerNorm = new LayerNorm(config.nEmbd);
        this.postAttentionLayerNorm = new LayerNorm(config.nEmbd);
        this.self_attn = new GPT2Attention(config);
        this.mlp = new GPT2MLP(config);
    }

    forward(params: Params, x: np.Array, mask: np.Array): np.Array {
        // Pre-LN attention with residual
        let residual = x;
        x = this.inputLayerNorm.forward(params.inputLayerNorm as Params, x);
        x = this.self_attn.forward(params.self_attn as Params, x, mask);
        x = x.add(residual);

        // Pre-LN MLP with residual
        residual = x;
        x = this.postAttentionLayerNorm.forward(params.postAttentionLayerNorm as Params, x);
        x = this.mlp.forward(params.mlp as Params, x);
        x = x.add(residual);
        return x;
    }
}

class GPT2 extends Module {
    config: GPT2Config;
    embed_tokens: Embed;
    embed_positions: Embed;
    blocks: GPT2Block[];
    lnF: LayerNorm;
    lmHead: Linear;

    constructor(config: GPT2Config) {
        super();
        this.config = config;

        this.embed_tokens = new Embed(config.vocabSize, config.nEmbd);
        this.embed_positions = new Embed(config.sequenceLen, config.nEmbd);

        this.blocks = [];
        for (let i = 0; i < config.nLayer; i++) {
            this.blocks.push(new GPT2Block(config));
        }

        this.lnF = new LayerNorm(config.nEmbd);
        this.lmHead = new Linear(config.nEmbd, config.vocabSize, false);
    }

    forward(params: Params, inputIds: np.Array): np.Array {
        const L = inputIds.shape[1];

        // Token + positional embeddings
        const positions = np.arange(L);
        let x = this.embed_tokens.forward(params.embed_tokens as Params, inputIds)
            .add(this.embed_positions.forward(params.embed_positions as Params, positions));

        // Causal mask: [1, 1, L, L]
        const mask = createCausalMask(L);

        // Transformer blocks
        const blockParams = params.blocks as Params[];
        for (let i = 0; i < this.blocks.length; i++) {
            x = this.blocks[i].forward(blockParams[i], x, mask);
        }

        // Final layer norm + LM head
        x = this.lnF.forward(params.lnF as Params, x);
        const logits = this.lmHead.forward(params.lmHead as Params, x);
        return logits;
    }
}

export { GPT2, GPT2Config };
