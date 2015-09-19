export function linear(x) {
  return x;
}

export function relu(x) {
  let y = [];
  for (let i = 0; i < x.length; i++) {
    y.push( (x[i] + Math.abs(x[i])) / 2.0 );
  }
  return y;
}

export function sigmoid(x) {
  let y = [];
  for (let i = 0; i < x.length; i++) {
    y.push( 1.0 / (1.0 + Math.exp(-x[i])) );
  }
  return y;
}

export function sigmoidHard(x) {
  let y = [];
  for (let i = 0; i < x.length; i++) {
    let y_i = x[i] * 0.2 + 0.5;
    if (y_i > 1) y_i = 1;
    if (y_i < 0) y_i = 0;
    y.push(y_i);
  }
  return y;
}

export function tanh(x) {
  let y = [];
  for (let i = 0; i < x.length; i++) {
    y.push( (Math.exp(2.0 * x[i] ) - 1) / (Math.exp(2.0 * x[i] ) + 1) );
  }
  return y;
}

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
