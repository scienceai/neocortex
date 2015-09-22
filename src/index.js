import * as activationFuncs from './functions/activations';
import { denseLayer } from './layers/dense';
import { embeddingLayer } from './layers/embedding';
import { batchNormalizationLayer } from './layers/normalization';
import { rGRULayer } from './layers/recurrent';

import { MatmulWebGL } from './lib/matmulWebGL';

import { NeuralNet } from './neuralnet';

export {
  activationFuncs,
  denseLayer,
  embeddingLayer,
  batchNormalizationLayer,
  rGRULayer,
  MatmulWebGL,
  NeuralNet
};
