export class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float64Array !== 'undefined') ? Float64Array : Array;
    this.USE_SIMD = (this.ARRAY_TYPE === Float64Array) && ('SIMD' in this);
    this.USE_WEBGL = false;

    this._layers = [];
  }

  addLayer(layer) {
    this._layers.push(layer);
  }

  predict(input) {

  }

}
