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
export function relu(x, useSIMD=false) {
  if (useSIMD) {
    // x isinstance of Float32Array
    let y = new Float32Array(x.length);
    let zeros = SIMD.Float32x4.splat(0);

    for (let i = 0; i < x.length; i += 4) {
      SIMD.Float32x4.store(y, i, SIMD.Float32x4.max(SIMD.Float32x4.load(x, i), zeros));
    }

    // returns an instance of Float32Array
    return y;

  } else {
    // x isinstance of Array
    let y = [];

    for (let i = 0; i < x.length; i++) {
      y.push( (x[i] + Math.abs(x[i])) / 2.0 );
    }

    // returns an instance of Array
    return y;
  }
}

/**
 * Sigmoid function
 */
export function sigmoid(x, useSIMD=false) {
  if (useSIMD) {
    // x isinstance of Float32Array
    let y = new Float32Array(x.length);
    let zeros = SIMD.Float32x4.splat(0);

    for(let i = 0; i < x.length; i += 4) {
      let x_f = SIMD.Float32x4.mul(SIMD.Float32x4(0.5, 0.5, 0.5, 0.5), SIMD.Float32x4.load(x, i));
      let mask1 = SIMD.Float32x4.greaterThanOrEqual(x_f, zeros);

      if x >= 0:
          if x < 1.7:
              z = (1.5 * x / (1 + x))
          elif x < 3:
              z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7))
          else:
              z = 0.99505475368673
      else:
          xx = -x
          if xx < 1.7:
              z = (1.5 * xx / (1 + xx))
          elif xx < 3:
              z = (0.935409070603099 + 0.0458812946797165 * (xx - 1.7))
          else:
              z = 0.99505475368673
          z = -z

      return 0.5 * (z + 1.)

      SIMD.Float32x4.store(y, i, SIMD.Float32x4.max(SIMD.Float32x4.load(x, i), zeros));
    }

    // returns an instance of Float32Array
    return y;
  } else {
    // x isinstance of Array
    let y = [];

    for (let i = 0; i < x.length; i++) {
      y.push( 1.0 / (1.0 + Math.exp(-x[i])) );
    }

    // returns an instance of Array
    return y;
  }
}

/**
 * Hard sigmoid function
 * approximate sigmoid with increased computational efficiency
 */
export function sigmoidHard(x, useSIMD=false) {
  if (useSIMD) {
    // x isinstance of Float32Array
    let y = new Float32Array(x.length);
    let zeros = SIMD.Float32x4.splat(0);
    let ones = SIMD.Float32x4.splat(1);
    let pointTwo = SIMD.Float32x4.splat(0.2);
    let pointFive = SIMD.Float32x4.splat(0.5);

    let y_i;
    for (let i = 0; i < x.length; i += 4) {
      y_i = SIMD.Float32x4.load(x, i);
      y_i = SIMD.Float32x4.add(SIMD.Float32x4.mul(y_i, pointTwo), pointFive);
      y_i = SIMD.Float32x4.max(SIMD.Float32x4.min(y_i, ones), zeros);
      SIMD.Float32x4.store(y, i, y_i);
    }

    // returns an instance of Float32Array
    return y;
  } else {
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
