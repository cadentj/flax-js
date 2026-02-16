import { numpy as np } from "@jax-js/jax";

class Module {
    training: boolean;

    constructor() {
        this.training = true;
    }

    forward(x: np.Array): np.Array {
        throw new Error("Subclasses must implement forward method");
    }
}

export { Module };