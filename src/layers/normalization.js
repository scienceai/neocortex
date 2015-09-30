import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';

const EPSILON = 0.000001;

export function batchNormalizationLayer(arrayType, x, weights, epsilon=EPSILON) {
  let gamma = pack(weights['gamma']);
  let beta = pack(weights['beta']);
  let mean = pack(weights['mean']);
  let std = pack(weights['std']);

  let y = ndarray(new arrayType(x.size), x.shape);

  ops.addseq(std, epsilon);
  ops.sub(y, x, mean);
  ops.addeq(ops.muleq(ops.diveq(y, std), gamma), beta);

  return y;
}
