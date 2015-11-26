import * as layerFuncs from './layers';
import ndarray from 'ndarray';
import pack from './lib/ndarray-pack';
import unpack from 'ndarray-unpack';
import request from 'axios';
import fs from 'fs';
import zlib from 'zlib';
import concat from 'concat-stream';
import Promise from 'bluebird';

export default class NeuralNet {
  constructor(config) {
    config = config || {};

    if (config.arrayType === 'float32') {
      this._arrayType = Float32Array;
    } else if (config.arrayType === 'float64') {
      this._arrayType = Float64Array;
    } else {
      this._arrayType = Array;
    }

    if (typeof window === 'object') {
      this._environment = 'browser';
    } else if (typeof importScripts === 'function') {
      this._environment = 'webworker';
    } else if (typeof process === 'object' && typeof require === 'function') {
      this._environment = 'node';
    } else {
      this._environment = 'shell';
    }
    console.log(`Neural network running in environment: ${this._environment}.`);

    this._modelFilePath = config.modelFilePath || null;
    this._layers = [];
  }

  init() {
    if (!this._modelFilePath) {
      throw new Error('no modelFilePath specified in config object.');
    }

    if (this._environment === 'browser' || this._environment === 'webworker') {
      return new Promise((resolve, reject) => {
        request.get(this._modelFilePath).then(res => {
          if (res.statusCode == 200) {
            this._layers = res.body;
            resolve(true);
          } else {
            reject(new Error('error loading model file.'));
          }
        }).catch(err => reject(err));
      });
    } else if (this._environment === 'node') {
      return new Promise((resolve, reject) => {
        let s = fs.createReadStream(this._modelFilePath);
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
      });
    }
  }

  predict(input) {
    let _predict = (X) => {
      for (let layer of this._layers) {
        let { layerName, parameters } = layer;
        X = layerFuncs[layerName](this._arrayType, X, ...parameters);
      }
      return X;
    };

    let X = pack(this._arrayType, input);
    let output_ndarray = _predict(X);

    return unpack(output_ndarray);
  }

}
