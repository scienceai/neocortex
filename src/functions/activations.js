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
  // x isinstance of Array
  let y = [];

  for (let i = 0; i < x.length; i++) {
    y.push( (x[i] + Math.abs(x[i])) / 2.0 );
  }

  // returns an instance of Array
  return y;
}

/**
 * Sigmoid function
 */
export function sigmoid(x) {
  // x isinstance of Array
  let y = [];

  for (let i = 0; i < x.length; i++) {
    y.push( 1.0 / (1.0 + Math.exp(-x[i])) );
  }

  // returns an instance of Array
  return y;
}

/**
 * Hard sigmoid function
 * approximate sigmoid with increased computational efficiency
 */
export function sigmoidHard(x) {
  // x isinstance of Array
  let y = [];

  for (let i = 0; i < x.length; i++) {
    let y_i = x[i] * 0.2 + 0.5;
    if (y_i > 1) y_i = 1;
    if (y_i < 0) y_i = 0;
    y.push(y_i);
  }

  // returns an instance of Array
  return y;
}

/**
 * Hyperbolic tangent
 */
export function tanh(x) {
  let y = [];
  for (let i = 0; i < x.length; i++) {
    y.push( (Math.exp(2.0 * x[i] ) - 1) / (Math.exp(2.0 * x[i] ) + 1) );
  }
  return y;
}

/**
 * Softmax function
 */
export function softmax(x) {
  let e = [];
  for (let i = 0; i < x.length; i++) {
    e.push( Math.exp(x[i]) );
  }
  let sum = e.reduce((a, b) => a + b);

  let y = [];
  for (let i = 0; i < e.length; i++) {
    y.push( e[i] / sum );
  }
  return y;
}
