export class NeuralNet {
  constructor(config) {
    config = config || {};

    this.ARRAY_TYPE = (typeof Float64Array !== 'undefined') ? Float64Array : Array;
    this.SIMD_AVAIL = (this.ARRAY_TYPE === Float64Array) && ('SIMD' in this);
    this.WEBGL_AVAIL = true;

    this.modelFile = config.modelFile || null;
    if (this.modelFile) {
      this.loadModel(this.modelFile);
    }

    this._layers = [];
  }

  loadModel(modelFile) {
    // loads model file in proper json format
    let model = require(os.path.join(__dirname, modelFile));
    this._layers = model;
  }

  addLayer(layerName, parameters) {
    this._layers.push({ layerName, parameters });
  }

  predict(input) {
    let X = input;

    for (let layer of this._layers) {
      let { layerName, parameters } = layer;

      X = layerFuncs[layerName](X, ...parameters);
    }

    return X;
  }

}
