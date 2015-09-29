import * as layerFuncs from './layers';
import ndarray from 'ndarray';
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
      let s = fs.createReadStream(__dirname + modelFilePath);
      if (this.modelFileType === 'gzip') {
        let gunzip = zlib.createGunzip();
        s.pipe(gunzip).pipe(concat((model) => {
          this._layers = JSON.parse(model.toString());
          this.readyStatus = true;
        }));
      } else if (this.modelFileType === 'json') {
        s.pipe(concat((model) => {
          this._layers = JSON.parse(model.toString());
          this.readyStatus = true;
        }));
      }
    } else if (this.environment === 'browser') {
      if (this.modelFileType === 'gzip') {
        request.get(modelFilePath)
          .set('Content-Type', 'application/json')
          .set('Content-Encoding', 'gzip')
          .end((err, res) => {
            if (err) return console.error('error loading model file.');
            if (res.statusCode == 200) {
              this._layers = res.body;
              this.readyStatus = true;
            } else {
              console.error('error loading model file.');
            }
          });
      } else if (this.modelFileType === 'json') {
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
  }

  predict(input, shape, callback) {

    let _predict = (X) => {
      for (let layer of this._layers) {
        let { layerName, parameters } = layer;
        X = layerFuncs[layerName](X, ...parameters);
      }
      return X;
    };

    let X = ndarray(new Float32Array(input), shape);

    if (!this.readyStatus) {
      let waitReady = setInterval(() => {
        if (this.readyStatus) {
          clearInterval(waitReady);
          let output = _predict(X);
          callback(null, output);
        }
      }, 10);
    }

  }

}
