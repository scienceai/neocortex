export class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float32Array !== 'undefined') ? Float32Array : Array;
    this.USE_SIMD = (this.ARRAY_TYPE === Float32Array) && ('SIMD' in this);
    this.N_PARALLEL = 1;

  }

}
