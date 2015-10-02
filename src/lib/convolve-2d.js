import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export default function(y, X, H) {
  if (X.shape[0] !== H.shape[0] || X.shape[1] !== H.shape[1]) {
    throw new Error('first two dimensions of X and H must be equal.');
  }

  let nb_filter = X.shape[0];
  let stack_size = X.shape[1];
  let steps_i = X.shape[2] - H.shape[2] + 1;
  let steps_j = X.shape[3] - H.shape[3] + 1;

  for (let i = 0; i < steps_i; i++) {
    for (let j = 0; j < steps_j; j++) {

      let matTemp = ndarray(new X.data.constructor(H.size), H.shape);
      ops.mul(matTemp,
        X.hi(nb_filter, stack_size, i + H.shape[2], j + H.shape[3]).lo(0, 0, i, j),
        H.step(1, 1, -1, -1)
      );

      for (let filter = 0; filter < nb_filter; filter++) {
        y.set(filter, i, j, ops.sum(matTemp.pick(filter, null, null, null)));
      }

    }
  }

  return y;
}
