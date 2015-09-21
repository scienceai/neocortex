import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import mvprod from 'ndarray-matrix-vector-product';
import * as activationFuncs from '../functions/activations';

export function rGRULayer(x, weights, activation='tanh', innerActivation='sigmoidHard') {
  let { W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h } = weights;

  let x_t = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp2 = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let z_t = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp_xz = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp_hz = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let r_t = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp_xr = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp_hr = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let h_t = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp_xh = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let temp_hh = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);
  let h_tm1 = ndarray(new Float64Array(x.shape[1]), [x.shape[1]]);

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    mvprod(temp_hz, W_hz.transpose(1, 0), h_tm1);
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, temp_hz);
    ops.addeq(z_t, b_z);
    activationFuncs[innerActivation](z_t);

    mvprod(temp_xr, W_xr.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_xr);
    ops.addeq(r_t, temp_hr);
    ops.addeq(r_t, b_r);
    activationFuncs[innerActivation](r_t);

    mvprod(temp_xh, W_xh.transpose(1, 0), x_t);
    ops.mul(temp, r_t, h_tm1);
    mvprod(temp_hh, W_hh.transpose(1, 0), temp);
    ops.assigns(h_t, 0);
    ops.addeq(h_t, temp_xh);
    ops.addeq(h_t, temp_hh);
    ops.addeq(h_t, b_h);
    activationFuncs[activation](h_t);

    ops.mul(temp, z_t, h_tm1);
    ops.assign(temp2, z_t);
    ops.addseq(ops.negeq(temp2), 1);
    ops.muleq(temp2, h_t);
    ops.addeq(temp2, temp);
    ops.assign(h_t, temp2);
  }

  ops.assigns(h_t, 0);
  for (let i = 0; i < x.shape[0]; i++) {
    ops.assign(x_t, x.pick(i, null));
    _step();
  }

  return h_t;
}
