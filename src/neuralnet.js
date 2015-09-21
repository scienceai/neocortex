export class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float64Array !== 'undefined') ? Float64Array : Array;
    this.USE_SIMD = (this.ARRAY_TYPE === Float64Array) && ('SIMD' in this);
    this.N_PARALLEL = 1;

  }

}
