import { Module } from "./module.ts";
import { numpy as np } from "@jax-js/jax";
import { layerNorm } from "../core";

class LayerNorm extends Module {
    scale: np.Array | null = null;
    bias: np.Array | null = null;
    useScale: boolean;
    useBias: boolean;
    eps: number;

    constructor(
      normalizedShape: number,
      useBias: boolean = true,
      useScale: boolean = true,
      eps: number = 1e-5,
    ) {
        super();
        this.useBias = useBias;
        this.useScale = useScale;
        this.eps = eps;
        if (useScale) {
            this.scale = np.ones([normalizedShape]);
        }
        if (useBias) {
            this.bias = np.zeros([normalizedShape]);
        }
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