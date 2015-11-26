import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from '../lib/ndarray-pack';

const EPSILON = 0.000001;

export function batchNormalizationLayer(arrayType, x, weights, epsilon=EPSILON) {
  let gamma = pack(arrayType, weights['gamma']);
  let beta = pack(arrayType, weights['beta']);
  let mean = pack(arrayType, weights['mean']);
  let std = pack(arrayType, weights['std']);

  let y = ndarray(new arrayType(x.size), x.shape);

  ops.addseq(std, epsilon);
  ops.sub(y, x, mean);
  ops.addeq(ops.muleq(ops.diveq(y, std), gamma), beta);

  return y;
}
