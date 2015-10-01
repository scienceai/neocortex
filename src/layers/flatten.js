import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export function flattenLayer(arrayType, x) {

  if (x.shape.length === 1) {
    return x;
  } else if (x.shape.length === 2) {

    let y = ndarray(new arrayType(x.size), [x.size]);

    for (let i = 0; i < x.shape[0]; i++) {
      ops.assign(y.hi((i+1) * x.shape[1]).lo(i * x.shape[1]), x.pick(i, null));
    }

    return y;

  } else if (x.shape.length === 3) {

    let y = ndarray(new arrayType(x.size), [x.size]);

    let offset_i = 0;
    for (let i = 0; i < x.shape[0]; i++) {
      offset_i = i * x.shape[1] * x.shape[2];
      for (let j = 0; j < x.shape[1]; j++) {
        ops.assign(y.hi((j+1)*x.shape[2] + offset_i).lo(j*x.shape[2] + offset_i), x.pick(i, j, null));
      }
    }

    return y;

  }
}
