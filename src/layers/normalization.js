import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';

const EPSILON = 0.000001;

export function batchNormalizationLayer(arrayType, x, weights, epsilon=EPSILON) {
  let gamma = pack(weights['gamma']);
  let beta = pack(weights['beta']);
  let mean = pack(weights['mean']);
  let std = pack(weights['std']);

  ops.addeq(ops.muleq(ops.diveq(ops.subeq(x, mean), ops.addseq(std, epsilon)), gamma), beta);

  return x;
}
