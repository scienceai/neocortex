import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export function dropoutLayer(arrayType, x, p=0.5) {

  let y = ndarray(new arrayType(x.size), x.shape);
  ops.muls(y, x, 1 - p);

  return y;
}
