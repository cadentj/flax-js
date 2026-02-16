// Combinators of modules, such as a Sequential

import { Module } from "./module.ts";
import { numpy as np } from "@jax-js/jax";

class Sequential extends Module {
    layers: Module[];

    constructor(layers: Module[]) {
        super();
        this.layers = layers;
    }

    forward(x: np.Array) {
        let output = this.layers[0].forward(x);
        for (let i = 1; i < this.layers.length; i++) {
            output = this.layers[i].forward(output);
        }
        return output;
    }
}

export { Sequential };