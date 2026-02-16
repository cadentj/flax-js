import { Module } from "./module.ts";
import { numpy as np } from "@jax-js/jax";
import { layerNorm } from "../core";

class LayerNorm extends Module {
    scale: np.Array;
    bias: np.Array;
    useScale: boolean;
    useBias: boolean;
    eps: number;

    constructor(
      useBias: boolean = true,
      useScale: boolean = true,
      eps: number = 1e-6,
    ) {
        super();
        this.useBias = useBias;
        this.useScale = useScale;
        this.eps = eps;
    }

    forward(x: np.Array): np.Array {
      return layerNorm(
        x, 
        this.bias,
        this.scale,
        this.eps,
      );
    }
}

export { LayerNorm };