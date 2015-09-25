let F32x4 = SIMD.Float32x4;
let B32x4 = SIMD.Bool32x4;
let I32x4 = SIMD.Int32x4;

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
  // x isinstance of Float32Array
  let y = new Float32Array(x.length);
  let zeros = F32x4.splat(0);

  for (let i = 0; i < x.length; i += 4) {
    F32x4.store(y, i, F32x4.max(F32x4.load(x, i), zeros));
  }

  // returns an instance of Float32Array
  return y;
}

/**
 * Sigmoid function
 */
export function sigmoid(x) {
  // x isinstance of Float32Array
  let y = new Float32Array(x.length);
  let zeros = F32x4.splat(0);

  for(let i = 0; i < x.length; i += 4) {
    let x_f = F32x4.mul(F32x4.splat(0.5), F32x4.load(x, i));
    let mask_0 = B32x4.and(F32x4.greaterThanOrEqual(x_f, zeros), F32x4.lessThan(x_f, F32x4.splat(1.7)));
    let mask_1 = B32x4.and(F32x4.greaterThanOrEqual(x_f, F32x4.splat(1.7)), F32x4.lessThan(x_f, F32x4.splat(3)));
    let mask_2 = F32x4.greaterThanOrEqual(x_f, F32x4.splat(3));
    let mask_3 = B32x4.and(F32x4.greaterThanOrEqual(x_f, F32x4.splat(-1.7)), F32x4.lessThan(x_f, zeros));
    let mask_4 = B32x4.and(F32x4.greaterThanOrEqual(x_f, F32x4.splat(-3)), F32x4.lessThan(x_f, F32x4.splat(-1.7)));
    let mask_5 = F32x4.lessThan(x_f, F32x4.splat(-3));

    x_f = F32x4.select(mask_0, F32x4.div(F32x4.mul(F32x4.splat(1.5), x_f), F32x4.add(F32x4.splat(1), x_f)), x_f);

    x_f = F32x4.select(mask_1, F32x4.add(F32x4.splat(0.935409070603099), F32x4.mul(F32x4.splat(0.0458812946797165), F32x4.sub(x_f, F32x4.splat(1.7)))), x_f);

    x_f = F32x4.select(mask_2, F32x4.splat(0.99505475368673), x_f);

    x_f = F32x4.select(mask_3, F32x4.div(F32x4.mul(F32x4.splat(1.5), x_f), F32x4.sub(F32x4.splat(1), x_f)), x_f);

    x_f = F32x4.select(mask_4, F32x4.add(F32x4.splat(-0.935409070603099), F32x4.mul(F32x4.splat(0.0458812946797165), F32x4.sub(x_f, F32x4.splat(1.7)))), x_f);

    x_f = F32x4.select(mask_5, F32x4.splat(-0.99505475368673), x_f);

    x_f = F32x4.mul(F32x4.splat(0.5), F32x4.add(x_f, F32x4.splat(1)));

    F32x4.store(y, i, x_f);
  }

  // returns an instance of Float32Array
  return y;
}

/**
 * Hard sigmoid function
 * approximate sigmoid with increased computational efficiency
 */
export function hard_sigmoid(x) {
  // x isinstance of Float32Array
  let y = new Float32Array(x.length);
  let zeros = F32x4.splat(0);
  let ones = F32x4.splat(1);
  let pointTwo = F32x4.splat(0.2);
  let pointFive = F32x4.splat(0.5);

  let y_i;
  for (let i = 0; i < x.length; i += 4) {
    y_i = F32x4.load(x, i);
    y_i = F32x4.add(F32x4.mul(y_i, pointTwo), pointFive);
    y_i = F32x4.max(F32x4.min(y_i, ones), zeros);
    F32x4.store(y, i, y_i);
  }

  // returns an instance of Float32Array
  return y;
}

/**
 * Hyperbolic tangent
 * TODO: implement SIMD
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
 * TODO: implement SIMD
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
