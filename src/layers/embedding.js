import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export function embeddingLayer(x, weights) {
  let { E } = weights;

  let nnz = x.data.reduce((a, b) => a + ((b > 0) ? 1 : 0), 0);
  let y = ndarray(new Float64Array(nnz * E.shape[1]), [nnz, E.shape[1]]);

  let i_nnz = 0;
  for (let i = 0; i < x.shape[0]; i++) {
    if (x.get(i) > 0) {
      ops.assign(y.pick(i_nnz, null), E.pick(x.get(i), null));
      i_nnz += 1;
    }
  }

  return y;
}
