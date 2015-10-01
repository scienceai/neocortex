import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export default function(y, X, H) {
  let steps_i = X.shape[0] - H.shape[0] + 1;
  let steps_j = X.shape[1] - H.shape[1] + 1;

  for (let i = 0; i < steps_i; i++) {
    for (let j = 0; j < steps_j; j++) {
      let matTemp = ndarray(new Float64Array(H.size), H.shape);
      ops.mul(matTemp, X.hi(i + H.shape[0], j + H.shape[1]).lo(i, j), H.step(-1, -1));
      y.set(i, j, ops.sum(matTemp));
    }
  }
  return y;
}
