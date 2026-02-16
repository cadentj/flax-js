import { Module } from "./module.ts";
import { numpy as np } from "@jax-js/jax";

class Linear extends Module {
    inFeatures: number;
    outFeatures: number;
    bias: np.Array | null = null;
    useBias: boolean;
    weight: np.Array;

    constructor(inFeatures: number, outFeatures: number, useBias: boolean = true) {
        super();
        
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        this.useBias = useBias;

        this.weight = np.zeros([outFeatures, inFeatures]);
        if (useBias) {
            this.bias = np.zeros([outFeatures]);
        }
    }

    forward(x: np.Array): np.Array {
        x = np.matmul(x, this.weight)
        if (this.useBias) {
            x = x.add(this.bias!);
        }
        return x;
    }
}

export { Linear };

class Embed extends Module {
    numEmbeddings: number;
    embedding: np.Array;

    constructor(numEmbeddings: number, embeddingDim: number) {
        super();
        this.numEmbeddings = numEmbeddings;
        this.embedding = np.zeros([numEmbeddings, embeddingDim]);
    }
    
    forward(inputs: np.Array): np.Array {
        if (inputs.dtype !== "int32") {
            throw new Error("Embed input must be of type int32");
        }
        return np.take(this.embedding, inputs, 0);
    }
}

export { Embed };