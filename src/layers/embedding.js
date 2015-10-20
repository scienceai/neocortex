import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';

export function embeddingLayer(arrayType, x, weights, mask_zero=false) {
  let E = pack(weights['E']);

  let y;

  if (mask_zero) {
    let nnz = x.data.reduce((a, b) => a + ((b > 0) ? 1 : 0), 0);

    y = ndarray(new arrayType(nnz * E.shape[1]), [nnz, E.shape[1]]);

    let i_nnz = 0;
    for (let i = 0; i < x.shape[0]; i++) {
      if (x.get(i) > 0) {
        ops.assign(y.pick(i_nnz, null), E.pick(x.get(i), null));
        i_nnz += 1;
      }
    }
  } else {

    y = ndarray(new arrayType(x.shape[0] * E.shape[1]), [x.shape[0], E.shape[1]]);

    for (let i = 0; i < x.shape[0]; i++) {
      ops.assign(y.pick(i, null), E.pick(x.get(i), null));
    }

  }

  return y;
}
