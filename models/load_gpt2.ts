import fs from "node:fs";
import { safetensors } from "@jax-js/loaders";

const buf = fs.readFileSync(".cache/model.safetensors");
const result = safetensors.parse(buf);
console.log(result);