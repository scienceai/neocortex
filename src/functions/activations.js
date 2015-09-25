import ops from 'ndarray-ops';

/**
 * Linear function
 * 1-to-1 input-output mapping
 */
export function linear(x) {
  return x;
}

/**
 * Rectified linear unit
 */
export function relu(x) {
  ops.divseq(ops.addeq(x, ops.abseq(x)), 2.0);
  return x;
}

/**
 * Sigmoid function
 */
export function sigmoid(x) {
  ops.recipeq(ops.addseq(ops.expeq(ops.negeq(x)), 1.0));
  return x;
}

/**
 * Hard sigmoid function
 * approximate sigmoid with increased computational efficiency
 */
export function hard_sigmoid(x) {
  ops.addseq(ops.mulseq(x, 0.2), 0.5);
  for (let i = 0; i < x.size; i++) {
    if (x.data[i] > 1) x.data[i] = 1;
    if (x.data[i] < 0) x.data[i] = 0;
  }
  return x;
}

/**
 * Hyperbolic tangent
 */
export function tanh(x) {
  ops.expeq(ops.mulseq(x, 2.0));
  for (let i = 0; i < x.size; i++) {
    x.data[i] = (x.data[i] - 1) / (x.data[i] + 1);
  }
  return x;
}

/**
 * Softmax function
 */
export function softmax(x) {
  ops.expeq(x);
  let sum = ops.sum(x);
  ops.divseq(x, sum);
  return x;
}
