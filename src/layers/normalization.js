import ndarray from 'ndarray';
import ops from 'ndarray-ops';

const EPSILON = 0.000001;

export function batchNormalization(x, weights) {
  let { gamma, beta, mean, std } = weights;

  ops.addeq(ops.muleq(ops.diveq(ops.subeq(x, mean), ops.addseq(std, EPSILON)), gamma), beta);

  return x;
}
