import cwise from 'cwise';
import ndarray from 'ndarray';
import pack from '../lib/ndarray-pack';
import ops from 'ndarray-ops';

export function leakyReLULayer(arrayType, x, alpha=0.3) {

  let leakyReLU = cwise({
    args: ['array', 'array', 'array', 'scalar'],
    body: function(_y, _x, _x_abs, _alpha) {
      _y = ((_x + _x_abs) / 2.0) + _alpha * ((_x - _x_abs) / 2.0);
    }
  });

  let y = ndarray(new arrayType(x.size), x.shape);

  let x_abs = ndarray(new arrayType(x.size), x.shape);
  ops.abs(x_abs, x);
  leakyReLU(y, x, x_abs, alpha);

  return y;
}

export function parametricReLULayer(arrayType, x, weights) {
  let alphas = pack(arrayType, weights['alphas']);

  let parametricReLU = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_y, _x, _x_abs, _alphas) {
      _y = ((_x + _x_abs) / 2.0) + _alphas * ((_x - _x_abs) / 2.0);
    }
  });

  let y = ndarray(new arrayType(x.size), x.shape);

  let x_abs = ndarray(new arrayType(x.size), x.shape);
  ops.abs(x_abs, x);
  parametricReLU(y, x, x_abs, alphas);

  return y;
}

export function parametricSoftplusLayer(arrayType, x, weights) {
  let alphas = pack(arrayType, weights['alphas']);
  let betas = pack(arrayType, weights['betas']);

  let y = ndarray(new arrayType(x.size), x.shape);

  ops.mul(y, x, betas);
  ops.muleq(ops.logeq(ops.addseq(ops.expeq(y), 1)), alphas);

  return y;
}

export function thresholdedLinearLayer(arrayType, x, theta=1.0) {

  let thresholdedLinear = cwise({
    args: ['array', 'array', 'array', 'scalar'],
    body: function(_y, _x, _x_abs, _theta) {
      _y = (_x_abs < _theta) ? 0 : _x;
    }
  });

  let y = ndarray(new arrayType(x.size), x.shape);

  let x_abs = ndarray(new arrayType(x.size), x.shape);
  ops.abs(x_abs, x);
  thresholdedLinear(y, x, x_abs, theta);

  return y;
}

export function thresholdedReLuLayer(arrayType, x, theta=1.0) {

  let thresholdedReLu = cwise({
    args: ['array', 'array', 'scalar'],
    body: function(_y, _x, _theta) {
      _y = (_x > _theta) ? _x : 0;
    }
  });

  let y = ndarray(new arrayType(x.size), x.shape);

  thresholdedReLu(y, x, theta);

  return y;
}
