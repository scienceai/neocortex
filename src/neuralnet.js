import * as layerFuncs from './layers';
import request from 'superagent';
import zlib from 'zlib';
import concat from 'concat-stream';

export default class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float64Array !== 'undefined') ? Float64Array : Array;
    this.SIMD_AVAIL = (this.ARRAY_TYPE === Float64Array) && ('SIMD' in this);
    this.WEBGL_AVAIL = true;

    if (typeof window !== 'undefined') {
      this.environment = 'browser';
    } else {
      this.environment = 'node';
    }

    this.readyStatus = false;
    this._layers = [];

    // json or gzip
    this.modelFileType = config.modelFileType || 'json';

    this.modelFilePath = config.modelFilePath || null;
    if (this.modelFilePath) {
      this.loadModel(this.modelFilePath);
    } else {
      throw new Error('no modelFilePath specified in config object.');
    }
  }

  loadModel(modelFilePath) {
    if (this.environment === 'node') {
      let gunzip = zlib.createGunzip();
      fs.createReadStream(__dirname + modelFilePath)
        .pipe(gunzip)
        .pipe(concat((model) => {
          this._layers = JSON.parse(model.toString());
          this.readyStatus = true;
        }));
    } else {
      request.get(modelFilePath)
        .set('Content-Type', 'application/json')
        .end((err, res) => {
          if (err) return console.error('error loading model file.');
          if (res.statusCode == 200) {
            this._layers = res.body;
            this.readyStatus = true;
          } else {
            console.error('error loading model file.');
          }
        });
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
