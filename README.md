# neuralnet-predict.js

Run trained neural networks in the browser or node.js.

The mission is to enable AI-powered web applications to be more easily deployed at scale by offloading as much computation as is reasonably possible to the client. While this library can be used within node.js, the real target here is the modern web browser. By focusing purely on prediction of already trained neural networks, we can take into full considerations the constraints of client hardware and the capabilities of browsers in their current state.

Given a neural network architecture and trained weights, the goal is to make forward predictive passes through the network as computationally efficient as possible. Computation on GPU is perfomed where possible, and parallelization on CPU is performed where possible.


### Tests

```sh
npm test
```

Tests for SIMD code uses a [SIMD shim/polyfill](https://github.com/ljharb/simd) based on the [ES7 proposal](https://github.com/tc39/ecmascript_simd). Currently only the firefox nightly build implements SIMD.
