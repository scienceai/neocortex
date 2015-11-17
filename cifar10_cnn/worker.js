'use strict';

importScripts('../neocortex.min.js');

let nn = new NeuralNet({
  modelFilePath: 'https://s3.amazonaws.com/neocortex-js/examples-data/cifar10_cnn_model_params.json.gz',
  arrayType: 'float32',
  useGPU: false
});

// save initial page onLoad messages
let onLoadMsgs = [];
self.onmessage = function(e) {
  onLoadMsgs.push(e);
};

nn.init().then(function() {

    function handleMsg(e) {

      // prediction given sample image
      let predictions = nn.predict(e.data.sampleData);
      postMessage({
        sampleNum: e.data.sampleNum,
        predictions: predictions,
        sampleLabel: e.data.sampleLabel
      });

    }

    // handle initial page onLoad messages
    onLoadMsgs.forEach(function(msg) {
      handleMsg(msg);
    });

    // handle additional messages
    self.onmessage = handleMsg;

}).catch(function(e) {
  console.error(e);
});
