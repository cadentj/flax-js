import { Module, Params } from "./module.ts";
import { numpy as np } from "@jax-js/jax";

class Linear extends Module {
    inFeatures: number;
    outFeatures: number;
    useBias: boolean;

    constructor(inFeatures: number, outFeatures: number, useBias: boolean = true) {
        super();
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = useBias;
    }

    initParams(): Params {
        return {
            weight: np.zeros([this.outFeatures, this.inFeatures]),
            ...(this.useBias ? { bias: np.zeros([this.outFeatures]) } : {}),
        };
    }

    forward(params: Params, x: np.Array): np.Array {
        x = np.matmul(x, params.weight as np.Array);
        if (this.useBias) {
            x = x.add(params.bias as np.Array);
        }
        return x;
    }
}

export { Linear };

class Embed extends Module {
    numEmbeddings: number;
    embeddingDim: number;

    constructor(numEmbeddings: number, embeddingDim: number) {
        super();
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
    }

    initParams(): Params {
        return { embedding: np.zeros([this.numEmbeddings, this.embeddingDim]) };
    }

    forward(params: Params, inputs: np.Array): np.Array {
        if (inputs.dtype !== "int32") {
            throw new Error("Embed input must be of type int32");
        }
        return np.take(params.embedding as np.Array, inputs, 0);
    }
}

export { Embed };
