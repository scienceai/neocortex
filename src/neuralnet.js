import * as layerFuncs from './layers';
import path from 'path';

export class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float64Array !== 'undefined') ? Float64Array : Array;
    this.SIMD_AVAIL = (this.ARRAY_TYPE === Float64Array) && ('SIMD' in this);
    this.WEBGL_AVAIL = true;

    this.readyStatus = false;
    this._layers = [];

    this.modelFile = config.modelFile || null;
    if (this.modelFile) {
      this.loadModel(this.modelFile);
    } else {
      throw new Error('no modelFile specified in config object.');
    }
  }

  loadModel(modelFile) {
    this._layers = require(`json!${path.join(__dirname, modelFile)}`);
    this.readyStatus = true;
  }

  predict(input) {
    let X = input;

    if (!this.readyStatus) {
      let waitReady = setInterval(() => {
        if (this.readyStatus) {
          clearInterval(waitReady);
        }
      }, 10);
    }

    for (let layer of this._layers) {
      let { layerName, parameters } = layer;

      X = layerFuncs[layerName](X, ...parameters);
    }

    return X;
  }

}
