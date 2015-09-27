import * as layerFuncs from './layers';

export default class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float64Array !== 'undefined') ? Float64Array : Array;
    this.SIMD_AVAIL = (this.ARRAY_TYPE === Float64Array) && ('SIMD' in this);
    this.WEBGL_AVAIL = true;

    if (typeof window === 'undefined') {
      this.environment = 'node';
    } else {
      this.environment = 'browser';
    }

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
    if (this.environment === 'node') {
      /*this._layers = require(path.join(__dirname, modelFile));*/
      this.readyStatus = true;
    } else {
      let req = new XMLHttpRequest();
      req.open('GET', modelFile, true);
      req.onreadystatechange = function (aEvt) {
        if (req.readyState == 4) {
           if(req.status == 200) {
             this._layers = JSON.parse(req.responseText);
             this.readyStatus = true;
           } else {
             console.error('cannot load model file.');
           }
        }
      };
      req.send(null);
    }
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
