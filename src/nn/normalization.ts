import { Module, Params } from "./module.ts";
import { numpy as np } from "@jax-js/jax";
import { layerNorm } from "../core";

class LayerNorm extends Module {
    normalizedShape: number;
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
        this.normalizedShape = normalizedShape;
        this.useBias = useBias;
        this.useScale = useScale;
        this.eps = eps;
    }

    initParams(): Params {
        return {
            ...(this.useScale ? { scale: np.ones([this.normalizedShape]) } : {}),
            ...(this.useBias ? { bias: np.zeros([this.normalizedShape]) } : {}),
        };
    }

    forward(params: Params, x: np.Array): np.Array {
        return layerNorm(
            x,
            (params.bias as np.Array) ?? null,
            (params.scale as np.Array) ?? null,
            this.eps,
        );
    }
}

export { LayerNorm };
