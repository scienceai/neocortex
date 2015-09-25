import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export function dropoutLayer(x, p=0.5) {

  ops.mulseq(x, p);

  return x;
}
