import fs from 'fs';
import zlib from 'zlib';
import Promise from 'bluebird';
import request from 'axios';
import ndarray from 'ndarray';
import pack from './lib/ndarray-pack';
import unpack from 'ndarray-unpack';
import * as layerFuncs from './layers';

let readFile = Promise.promisify(fs.readFile);
let gunzip = Promise.promisify(zlib.gunzip);

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

    this._modelFilePath = config.modelFilePath || null;
    this._layers = [];
  }

  init() {
    if (!this._modelFilePath) {
      throw new Error('no modelFilePath specified in config object.');
    }

    if (this._environment === 'browser' || this._environment === 'webworker') {
      return request.get(this._modelFilePath)
        .then(res => {
          if (res.status == 200) {
            this._layers = res.data;
          } else {
            throw new Error('error loading model file.');
          }
        })
        .catch(err => { throw err; });
    } else if (this._environment === 'node') {
      if (this._modelFilePath.endsWith('.json.gz')) {
        return Promise.resolve(this._modelFilePath)
          .then(readFile)
          .then(gunzip)
          .then(JSON.parse)
          .then(data => { this._layers = data; })
          .catch(err => { throw err; });
      } else if (this._modelFilePath.endsWith('.json')) {
        return Promise.resolve(this._modelFilePath)
          .then(readFile)
          .then(JSON.parse)
          .then(data => { this._layers = data; })
          .catch(err => { throw err; });
      }
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
