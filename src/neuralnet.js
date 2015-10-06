import * as layerFuncs from './layers';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import unpack from 'ndarray-unpack';
import request from 'superagent';
import zlib from 'zlib';
import concat from 'concat-stream';
import Promise from 'bluebird';

export default class NeuralNet {
  constructor(config) {
    config = config || {};

    this._arrayType = Float64Array || Array;
    if (config.arrayType === 'float32') {
      this._arrayType = Float32Array;
    } else if (config.arrayType === 'float64') {
      this._arrayType = Float64Array;
    }

    this._SIMD_AVAIL = (this._arrayType === Float32Array || this._arrayType === Float64Array) && ('SIMD' in this);
    this._WEBGL_AVAIL = true;

    this.useGPU = (config.useGPU || false) && this._WEBGL_AVAIL;

    if (typeof window === 'object') {
      this._environment = 'browser';
    } else if (typeof importScripts === 'function') {
      this._environment = 'webworker';
    } else if (typeof process === 'object' && typeof require === 'function') {
      this._environment = 'node';
    } else {
      this._environment = 'shell';
    }
    console.log(`Running from environment: ${this._environment}`);

    this._layers = [];

    this._modelFilePath = config.modelFilePath || null;
    this._sampleDataPath = config.sampleDataPath || null;
  }

  loadModel() {
    return new Promise((resolve, reject) => {
      if (!this._modelFilePath) {
        reject(new Error('no modelFilePath specified in config object.'));
      }

      if (this._environment === 'browser' || this._environment === 'webworker') {
        request.get(this._modelFilePath)
          .end((err, res) => {
            if (err) reject(err);
            if (res.statusCode == 200) {
              this._layers = res.body;
              resolve(true);
            } else {
              reject(new Error('error loading model file.'));
            }
          });
      } else if (this._environment === 'node') {
        let s = fs.createReadStream(__dirname + this._modelFilePath);
        if (this._modelFilePath.endsWith('.json.gz')) {
          let gunzip = zlib.createGunzip();
          s.pipe(gunzip).pipe(concat((model) => {
            this._layers = JSON.parse(model.toString());
            resolve(true);
          }));
        } else if (this._modelFilePath.endsWith('.json')) {
          s.pipe(concat((model) => {
            this._layers = JSON.parse(model.toString());
            resolve(true);
          }));
        }
      }
    });
  }

  loadSampleData() {
    return new Promise((resolve, reject) => {
      if (!this._sampleDataPath) {
        reject(new Error('no sampleDataPath specified in config object.'));
      }

      if (this._environment === 'node') {
        let s = fs.createReadStream(__dirname + this._sampleDataPath);
        if (this._sampleDataPath.endsWith('.json.gz')) {
          let gunzip = zlib.createGunzip();
          s.pipe(gunzip).pipe(concat((data) => {
            this.SAMPLE_DATA = JSON.parse(data.toString());
            resolve(true);
          }));
        } else if (this._sampleDataPath.endsWith('.json')) {
          s.pipe(concat((data) => {
            this.SAMPLE_DATA = JSON.parse(data.toString());
            resolve(true);
          }));
        }
      } else if (this._environment === 'browser' || this._environment === 'webworker') {
        request.get(this._sampleDataPath)
          .end((err, res) => {
            if (err) reject(err);
            if (res.statusCode == 200) {
              this.SAMPLE_DATA = res.body;
              resolve(true);
            } else {
              reject(new Error('error loading data file.'));
            }
          });
      }
    });
  }

  predict(input) {
    return new Promise((resolve, reject) => {

      let _predict = (X) => {
        for (let layer of this._layers) {
          let start = new Date().getTime();
          let { layerName, parameters } = layer;
          X = layerFuncs[layerName](this._arrayType, X, ...parameters);
          console.log(layerName, new Date().getTime() - start, X);
        }
        return X;
      };

      let X = pack(input);
      let output_ndarray = _predict(X);

      resolve(unpack(output_ndarray));

    });
  }

}
