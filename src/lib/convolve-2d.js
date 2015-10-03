import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import fft from 'ndarray-fft';
import pool from 'typedarray-pool';
import cwise from 'cwise';

let cmuleq = cwise({
  args: ['array', 'array', 'array', 'array'],
  body: function cmuleq(_out_r, _out_i, _a_r, _a_i) {
    var a = _a_r;
    var b = _a_i;
    var c = _out_r;
    var d = _out_i;
    var k1 = c * (a + b);
    _out_r = k1 - b * (c + d);
    _out_i = k1 + a * (d - c);
  }
});

function nextPow2(v) {
  v += v === 0;
  --v;
  v |= v >>> 1;
  v |= v >>> 2;
  v |= v >>> 4;
  v |= v >>> 8;
  v |= v >>> 16;
  return v + 1;
}

/****************************************************
* convolution implementation using FFT
*
* - A has dimensions of stack_size x rows_a x cols_a
* - H has dimensions of stack_size x rows_h x cols_h
* - out is summed over the stacks and thus has dimensions of
*   (rows_a - rows_h + 1) x (cols_a - cols_h + 1)
*/
export default function(out, A, H) {

  const d = 3;
  let nsize = 1;
  let nstride = new Array(d);
  let nshape = new Array(d);

  for (let i = d - 1; i >= 0; --i) {
    nshape[i] = nextPow2(A.shape[i] + H.shape[i] - 1);
    nstride[i] = nsize;
    nsize *= nshape[i];
  }

  let x_t = pool.mallocDouble(nsize);
  let x = ndarray(x_t, nshape, nstride, 0);
  ops.assigns(x, 0);
  ops.assign(x.hi.apply(x, A.shape), A);

  let y_t = pool.mallocDouble(nsize);
  let y = ndarray(y_t, nshape, nstride, 0);
  ops.assigns(y, 0);

  fft(1, x, y);

  let u_t = pool.mallocDouble(nsize);
  let u = ndarray(u_t, nshape, nstride, 0);
  ops.assigns(u, 0);
  ops.assign(u.hi.apply(u, H.shape), H);

  let v_t = pool.mallocDouble(nsize);
  let v = ndarray(v_t, nshape, nstride, 0);
  ops.assigns(v, 0);

  fft(1, u, v);
  cmuleq(x, y, u, v);
  fft(-1, x, y);

  /*4,8,8
  2,2,2
  2,4,4
  1,1,1

  4,8,8
  2.4,3
  2,4,4
  1,3,2*/

  let outTemp = ndarray(new A.data.constructor(A.size), A.shape);
  let out_offset = new Array(d);
  for (let i = d - 1; i >= 0; --i) {
    out_offset[i] = H.shape[i] - 1;
  }

  let cropped_x = x.lo.apply(x, out_offset);
  cropped_x = cropped_x.hi.apply(cropped_x, A.shape);
  ops.assign(outTemp, cropped_x);

  let out_rows = out.shape[0];
  let out_cols = out.shape[1];
  ops.assign(out, outTemp.pick(0, null, null).hi(out_rows, out_cols));

  pool.freeDouble(x_t);
  pool.freeDouble(y_t);
  pool.freeDouble(u_t);
  pool.freeDouble(v_t);
}

/*import ndarray from 'ndarray';
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
}*/
