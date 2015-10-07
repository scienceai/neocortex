import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import mvprod from '../lib/cpu/matrix-vector-product';
import * as activationFuncs from '../functions/activations';

export function denseLayer(arrayType, x, weights, activation='linear') {
  let W = pack(weights['W']);
  let b = pack(weights['b']);

  let y = ndarray(new arrayType(W.shape[1]), [W.shape[1]]);

  // W*x
  mvprod(y, W.transpose(1, 0), x);
  // W*x + b
  ops.addeq(y, b);
  // activation(W*x + b)
  activationFuncs[activation](y);

  return y;
}
