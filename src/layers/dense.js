import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import mvprod from 'ndarray-matrix-vector-product';
import * as activationFuncs from '../functions/activations';

export function denseLayer(arrayType, x, weights, activation='linear') {
  let W = pack(weights['W']);
  let b = pack(weights['b']);

  let y = ndarray(new arrayType(W.shape[0]), [W.shape[0]]);

  // W*x
  mvprod(y, W, x);
  // W*x + b
  ops.addeq(y, b);
  // activation(W*x + b)
  activationFuncs[activation](y);

  return y;
}
