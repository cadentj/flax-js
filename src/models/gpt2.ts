import { numpy as np } from "@jax-js/jax";
import { nn } from "@jax-js/jax";
import { Linear, LayerNorm, Module, Sequential } from "../nn";

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
    nKvHead: number;
    nEmbd: number;
    windowPattern: string;

    constructor(
        sequenceLen: number = 2048,
        vocabSize: number = 32768,
        nLayer: number = 12,
        nHead: number = 6,
        nKvHead: number = 6,
        nEmbd: number = 768,
        windowPattern: string = "SSSL",
    ) {
        this.sequenceLen = sequenceLen;
        this.vocabSize = vocabSize;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nKvHead = nKvHead;
        this.nEmbd = nEmbd;
        this.windowPattern = windowPattern;
    }
}

class GPT2Attention extends Module {
    layerIndex: number;
    nHead: number;
    nKvHead: number;
    nEmbd: number;
    headDim: number;
    qProj: Linear;
    kProj: Linear;
    vProj: Linear;
    oProj: Linear;
    veGateChannels: number;
    veGate: Linear | null;

    constructor(config: GPT2Config, layerIndex: number) {
        super();

        this.layerIndex = layerIndex;
        this.nHead = config.nHead;
        this.nKvHead = config.nKvHead;
        this.nEmbd = config.nEmbd;
        this.headDim = this.nEmbd / this.nHead;
        this.qProj = new Linear(this.nEmbd, this.nHead * this.headDim);
        this.kProj = new Linear(this.nEmbd, this.nKvHead * this.headDim);
        this.vProj = new Linear(this.nEmbd, this.nKvHead * this.headDim);
        this.oProj = new Linear(this.nEmbd, this.nEmbd);
    }

    forward(x_BD: np.Array) {

        let q_BH = this.qProj.forward(x_BD)
        let k_BH = this.kProj.forward(x_BD)
        let v_BH = this.vProj.forward(x_BD)

        return x_BD;
    }
}


class GPT2MLP extends Module {
    upProj: Linear;
    downProj: Linear;
    // act_fn

    constructor(config: GPT2Config) {
        super();
        this.upProj = new Linear(config.nEmbd, config.nEmbd * 4);
        this.downProj = new Linear(config.nEmbd * 4, config.nEmbd);
    }

    forward(x_BD: np.Array) {
        let x_BF = this.upProj.forward(x_BD)
        x_BF = nn.gelu(x_BF)
        x_BF = this.downProj.forward(x_BF)
        return x_BF
    }
}

class GPT2Block extends Module {
    attention: GPT2Attention;
    mlp: GPT2MLP;
    inputLayerNorm: LayerNorm;
    postAttentionLayerNorm: LayerNorm;

    constructor(config: GPT2Config) {
        super();
        this.attention = new GPT2Attention(config, 0);
        this.mlp = new GPT2MLP(config);
    }

    forward(x_BD: np.Array) {
        let residual = x_BD;
        x_BD = this.inputLayerNorm.forward(x_BD)
        x_BD = this.attention.forward(x_BD)
        x_BD = this.postAttentionLayerNorm.forward(x_BD)
        x_BD = x_BD.add(residual)
        x_BD = this.mlp.forward(x_BD)
        x_BD = x_BD.add(residual)
        return x_BD
    }
}

class GPT2 extends Module {
    config: GPT2Config;
    layers: Sequential;

    constructor(config: GPT2Config) {
        super();
        this.config = config;

        let _layers: GPT2Block[] = [];
        for (let i = 0; i < config.nLayer; i++) {
            _layers.push(new GPT2Block(config));
        }
        this.layers = new Sequential(_layers);
    }

    forward(x_BD: np.Array) {
        x_BD = this.layers.forward(x_BD)
        return x_BD;
    }
}

export { GPT2 };