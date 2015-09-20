import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import mvprod from 'ndarray-matrix-vector-product';
import { elemAdd } from '../lib/helper';
import * as activationFuncs from '../functions/activations';

export function denseLayer(x, W, b, activation='linear') {
  let y = ndarray(new Float32Array(W.shape[0]), [W.shape[0]]);

  // W*x
  mvprod(y, W, x);
  // W*x + b
  elemAdd(y, b);
  // activation(W*x + b)
  y.data = activationFuncs[activation](y.data);

  return y;
}
