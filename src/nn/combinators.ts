// Combinators of modules, such as a Sequential

import { Module, Params } from "./module.ts";
import { numpy as np } from "@jax-js/jax";

class Sequential extends Module {
    layers: Module[];

    constructor(layers: Module[]) {
        super();
        this.layers = layers;
    }

    forward(params: Params, x: np.Array): np.Array {
        const layerParams = params.layers as Params[];
        for (let i = 0; i < this.layers.length; i++) {
            x = this.layers[i].forward(layerParams[i], x);
        }
        return x;
    }
}

export { Sequential };
