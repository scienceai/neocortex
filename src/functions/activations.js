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
  let y = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    y[i] = (x[i] + Math.abs(x[i])) / 2.0;
  }

  return y;
}

/**
 * Sigmoid function
 */
export function sigmoid(x) {
  let y = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    y[i] = 1.0 / (1.0 + Math.exp(-x[i]));
  }

  return y;
}

/**
 * Hard sigmoid function
 * approximate sigmoid with increased computational efficiency
 */
export function sigmoidHard(x) {
  let y = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    let y_i = x[i] * 0.2 + 0.5;
    if (y_i > 1) y_i = 1;
    if (y_i < 0) y_i = 0;
    y[i] = y_i;
  }

  return y;
}

/**
 * Hyperbolic tangent
 */
export function tanh(x) {
  let y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    y[i] = (Math.exp(2.0 * x[i] ) - 1) / (Math.exp(2.0 * x[i] ) + 1);
  }
  return y;
}

/**
 * Softmax function
 */
export function softmax(x) {
  let e = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    e[i] = Math.exp(x[i]);
  }
  let sum = e.reduce((a, b) => a + b);

  let y = new Float32Array(e.length);
  for (let i = 0; i < e.length; i++) {
    y[i] = e[i] / sum;
  }
  return y;
}
