import * as layerFuncs from './layers';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import request from 'superagent';
import zlib from 'zlib';
import concat from 'concat-stream';

export default class NeuralNet {
  constructor(config) {
    config = config || {};

    this.arrayType = Float64Array || Array;
    if (config.arrayType === 'float32') {
      this.arrayType = Float32Array;
    } else if (config.arrayType === 'float64') {
      this.arrayType = Float64Array;
    }

    this._SIMD_AVAIL = (this.arrayType === Float32Array || this.arrayType === Float64Array) && ('SIMD' in this);
    this._WEBGL_AVAIL = true;

    this.useGPU = (config.useGPU || false) && this._WEBGL_AVAIL;

    if (typeof window !== 'undefined') {
      this._environment = 'browser';
    } else {
      this._environment = 'node';
    }

    this.readyStatus = false;
    this._layers = [];

    if (config.modelFilePath) {
      this.loadModel(config.modelFilePath);
    } else {
      throw new Error('no modelFilePath specified in config object.');
    }

    this.sampleDataLoaded = false;

    if (config.sampleDataPath) {
      this.loadSampleData(config.sampleDataPath);
    }
  }

  loadModel(modelFilePath) {
    if (this._environment === 'node') {
      let s = fs.createReadStream(__dirname + modelFilePath);
      if (modelFilePath.endsWith('.json.gz')) {
        let gunzip = zlib.createGunzip();
        s.pipe(gunzip).pipe(concat((model) => {
          this._layers = JSON.parse(model.toString());
          this.readyStatus = true;
        }));
      } else if (modelFilePath.endsWith('.json')) {
        s.pipe(concat((model) => {
          this._layers = JSON.parse(model.toString());
          this.readyStatus = true;
        }));
      }
    } else if (this._environment === 'browser') {
      request.get(modelFilePath)
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

  loadSampleData(sampleDataPath) {
    if (this._environment === 'node') {
      let s = fs.createReadStream(__dirname + sampleDataPath);
      if (sampleDataPath.endsWith('.json.gz')) {
        let gunzip = zlib.createGunzip();
        s.pipe(gunzip).pipe(concat((data) => {
          this.SAMPLE_DATA = JSON.parse(data.toString());
          this.sampleDataLoaded = true;
        }));
      } else if (sampleDataPath.endsWith('.json')) {
        s.pipe(concat((data) => {
          this.SAMPLE_DATA = JSON.parse(data.toString());
          this.sampleDataLoaded = true;
        }));
      }
    } else if (this._environment === 'browser') {
      request.get(sampleDataPath)
        .end((err, res) => {
          if (err) return console.error('error loading data file.');
          if (res.statusCode == 200) {
            this.SAMPLE_DATA = res.body;
            this.sampleDataLoaded = true;
          } else {
            console.error('error loading data file.');
          }
        });
    }
  }

  predict(input, callback) {

    let _predict = (X) => {
      for (let layer of this._layers) {
        let start = new Date().getTime();
        let { layerName, parameters } = layer;
        X = layerFuncs[layerName](this.arrayType, X, ...parameters);
        console.log(layerName, new Date().getTime() - start, X);
      }
      return X;
    };

    let X = pack(input);

    if (!this.readyStatus) {
      let waitReady = setInterval(() => {
        if (this.readyStatus) {
          clearInterval(waitReady);
          let output = _predict(X);
          callback(null, output);
        }
      }, 10);
    } else {
      let output = _predict(X);
      callback(null, output);
    }

  }

}
