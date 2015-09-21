# neuralnet-predict.js

Run trained neural networks in the browser or node.js.

Training deep neural networks on any meaningful dataset requires massive computational resources and lots and lots of time. However, prediction is relatively cheap - there is no backpropagation, loss functions, or optimization algorithms to worry about or that needs to be implemented (online learning is a different story). Chances are, your neural network is not going to be trained in javascript given its computational limitations, but you _do_ want your neural network to power a part of your client-facing web application. You either deploy your model on a server and call it from your web application through an API, or you can deploy it _in the browser_ alongside the rest of your webapp, with computation offloaded entirely to your end-user.

Perhaps most users will not be able to run billion-parameter networks in their browsers quite yet, but smaller networks are certainly within the realm of possibility.

By focusing purely on prediction of already trained neural networks, we can take into full considerations the constraints of client hardware and the capabilities of browsers in their current state. Given a neural network architecture and pre-trained weights, we would make forward predictive passes through the network as computationally efficient as possible. Computation on GPU is perfomed where possible, and parallelization on CPU is performed where possible.


### Usage


### Build


### Tests

```sh
npm test
```

Note: tests for SIMD code uses a [shim/polyfill](https://github.com/ljharb/simd) based on the [ES7 proposal](https://github.com/tc39/ecmascript_simd). The only browser that currently support SIMD is firefox nightly build.

### Colophon
