import { numpy as np } from "@jax-js/jax";

type Params = { [key: string]: np.Array | Params | Params[] };

class Module {
    training: boolean;

    constructor() {
        this.training = true;
    }

    initParams(): Params {
        return {};
    }

    init(): Params {
        const params = this.initParams();
        for (const [key, value] of Object.entries(this)) {
            if (value instanceof Module) {
                params[key] = value.init();
            } else if (Array.isArray(value) && value.length > 0 && value[0] instanceof Module) {
                params[key] = (value as Module[]).map(m => m.init());
            }
        }
        return params;
    }

    forward(params: Params, ...args: unknown[]): np.Array {
        throw new Error("Subclasses must implement forward method");
    }

    apply(params: Params, ...args: unknown[]): np.Array {
        return this.forward(params, ...args);
    }
}

export { Module, Params };
