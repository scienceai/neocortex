# neuralnet-predict.js

Run trained neural networks in the browser or node.js.

### Background

Training deep neural networks on any meaningful dataset requires massive computational resources and lots and lots of time. However, prediction is relatively cheap - there is no backpropagation, computational graphs, loss functions, or optimization algorithms to worry about or that needs to be implemented (online learning is a different story). Chances are, your neural network is not going to be trained in javascript given its computational limitations, but what if you _do_ want your neural network to power a part of your client-facing web application? You either deploy your model on a server and call it from your web application through an API, or you can deploy it _in the browser_ alongside the rest of your webapp, with computation offloaded entirely to your end-user.

A pipe dream? Perhaps most users will not be able to run billion-parameter networks in their browsers quite yet, but smaller networks are certainly within the realm of possibility!

By focusing purely on prediction of already trained neural networks, we can take into full considerations the constraints of client hardware and the capabilities of browsers in their current state. Given a neural network architecture and pre-trained weights, we would make forward predictive passes through the network as computationally efficient as possible. Computation on GPU is perfomed where possible and advantageous to do so, currently implemented using WebGL (hopefully WebCL will in the future make things more interesting).

What about using emscripten to compile models to be able to run in the browser? It may be possible (especially with the coming of WebAssembly), but it can be quite heavy and not trivial at all (see Cyrille Rosant's [attempts](http://cyrille.rossant.net/numpy-browser-llvm/) at running numpy in the browser through Numba and LLVM). By only focusing only on prediction, we can be as light and minimal as possible. The goal is to be able to serialize a Keras or Caffe model together with pretrained weights into javascript and be off and running.

Additionally, Andrej Karpathy's [ConvNetJS](https://github.com/karpathy/convnetjs) needs to be cited as a source of inspiration, as well as the excellent Theano-based deep learning framework [Keras](https://github.com/fchollet/keras/).


### Usage


### API

Functions and layers currently implemented are listed below. More forthcoming.

##### Activation functions

+ `linear`

+ `relu` (rectified linear or ReLU)

+ `sigmoid` or `sigmoidHard`

+ `tanh`

+ `softmax`

##### Dense fully-connected layers

+ `denseLayer`

##### Recurrent layers

+ `rGRULayer` (gated-recurrent unit or GRU)

+ `rLSTMLayer` (long short-term memory or LSTM)

+ `rJZS1Layer`, `rJZS2Layer`, `rJZS3Layer` (mutated GRUs - JZS1, JZS2, JZS3 - from [Jozefowicz et al. 2015](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf))

##### Convolutional layers

+ `convolution1DLayer`

+ `convolution2DLayer`

+ `pooling1DLayer`

+ `pooling2DLayer`

##### Embedding layers

+ `embeddingLayer` - maps indices to corresponding embedding vectors

##### Normalization layers

+ `batchNormalizationLayer` - see [Ioffe and Szegedy 2015](http://arxiv.org/abs/1502.03167)

### Build

```
$ npm run build
```

##### Webpack



### Tests

```
$ npm test
```

Note: tests for SIMD code uses a [shim/polyfill](https://github.com/ljharb/simd) based on the [ES7 proposal](https://github.com/tc39/ecmascript_simd). The only browser that currently support SIMD is firefox nightly build.

Browser testing is planned.

### Colophon

&copy; 2015 Leon Chen. MIT License.
