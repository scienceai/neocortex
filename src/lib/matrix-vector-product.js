import blas1 from 'ndarray-blas-level1';

// y = A*x
export default function(y, A, x) {
  for (let i = 0; i < A.shape[0]; i++) {
    y.set(i, blas1.dot(A.pick(i, null), x));
  }
  return y;
}
