'use strict';

importScripts('/neuralnet-predict.min.js');
importScripts('/bluebird.min.js');

let nn = new NeuralNet({
  modelFilePath: '/cifar10_cnn/cifar10_cnn_model_params.json',
  sampleDataPath: '/cifar10_cnn/sample_data.json',
  arrayType: 'float32',
  useGPU: false
});

// save initial page onLoad messages
let onLoadMsgs = [];
self.onmessage = function(e) {
  onLoadMsgs.push(e);
};

Promise.all([nn.loadModel(), nn.loadSampleData()])
  .then(function() {

    function handleMsg(e) {

      // prediction given sample image
      nn.predict(e.data.sampleData).then(function(predictions) {
        postMessage({
          sampleNum: e.data.sampleNum,
          predictions: predictions,
          sampleLabel: e.data.sampleLabel
        });
      });
    }

    // handle initial page onLoad messages
    onLoadMsgs.forEach(function(msg) {
      handleMsg(msg);
    });

    // handle additional messages
    self.onmessage = handleMsg;

  })
  .catch(function(e) {
    console.error(e);
  });
