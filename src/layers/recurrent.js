import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import mvprod from 'ndarray-matrix-vector-product';
import * as activationFuncs from '../functions/activations';

///////////////////////////////////////////////////////
// LSTM
//
export function rLSTMLayer(arrayType, x, weights, activation='tanh', innerActivation='sigmoidHard') {
  let { W_xi, W_hi, b_i, W_xc, W_hc, b_c, W_xf, W_hf, b_f, W_xo, W_ho, b_o } = weights;

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  let i_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xi = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hi = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let f_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xf = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hf = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let o_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xo = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_ho = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  let c_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xc = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hc = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_fc = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let c_tm1 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  let h_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_tm1 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xi, W_xi.transpose(1, 0), x_t);
    mvprod(temp_hi, W_hi.transpose(1, 0), h_tm1);
    ops.assigns(i_t, 0);
    ops.addeq(i_t, temp_xi);
    ops.addeq(i_t, temp_hi);
    ops.addeq(i_t, b_i);
    activationFuncs[innerActivation](i_t);

    mvprod(temp_xf, W_xf.transpose(1, 0), x_t);
    mvprod(temp_hf, W_hf.transpose(1, 0), h_tm1);
    ops.assigns(f_t, 0);
    ops.addeq(f_t, temp_xf);
    ops.addeq(f_t, temp_hf);
    ops.addeq(f_t, b_f);
    activationFuncs[innerActivation](f_t);

    mvprod(temp_xo, W_xo.transpose(1, 0), x_t);
    mvprod(temp_ho, W_ho.transpose(1, 0), h_tm1);
    ops.assigns(o_t, 0);
    ops.addeq(o_t, temp_xo);
    ops.addeq(o_t, temp_ho);
    ops.addeq(o_t, b_o);
    activationFuncs[innerActivation](o_t);

    mvprod(temp_xc, W_xc.transpose(1, 0), x_t);
    mvprod(temp_hc, W_hc.transpose(1, 0), h_tm1);
    ops.assigns(c_t, 0);
    ops.addeq(c_t, temp_xc);
    ops.addeq(c_t, temp_hc);
    ops.addeq(c_t, b_c);
    activationFuncs[activation](c_t);
    ops.muleq(c_t, i_t);
    ops.mul(temp_fc, f_t, c_tm1);
    ops.addeq(c_t, temp_fc);

    ops.assign(c_tm1, c_t);
    ops.mul(h_t, o_t, activationFuncs[activation](c_t));
  }

  ops.assigns(h_t, 0);
  for (let i = 0; i < x.shape[0]; i++) {
    ops.assign(x_t, x.pick(i, null));
    _step();
  }

  return h_t;
}


///////////////////////////////////////////////////////
// GRU
//
export function rGRULayer(arrayType, x, weights, activation='tanh', innerActivation='sigmoidHard') {
  let { W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h } = weights;

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp2 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let z_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let r_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_tm1 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

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


///////////////////////////////////////////////////////
// JZS1
//
export function rJZS1Layer(arrayType, x, weights, activation='tanh', innerActivation='sigmoid') {
  let { W_xz, b_z, W_xr, W_hr, b_r, W_hh, b_h, Pmat } = weights;

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp2 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let z_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let r_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_x_proj = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_tm1 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, b_z);
    activationFuncs[innerActivation](z_t);

    mvprod(temp_xr, W_xr.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_xr);
    ops.addeq(r_t, temp_hr);
    ops.addeq(r_t, b_r);
    activationFuncs[innerActivation](r_t);

    mvprod(temp_x_proj, Pmat.transpose(1, 0), x_t);
    ops.mul(temp, r_t, h_tm1);
    mvprod(temp_hh, W_hh.transpose(1, 0), temp);
    ops.assigns(h_t, 0);
    ops.addeq(h_t, activationFuncs['tanh'](temp_x_proj));
    ops.addeq(h_t, temp_hh);
    ops.addeq(h_t, b_h);
    activationFuncs[activation](h_t);

    ops.mul(temp, z_t, h_t);
    ops.assign(temp2, z_t);
    ops.addseq(ops.negeq(temp2), 1);
    ops.muleq(temp2, h_tm1);
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


///////////////////////////////////////////////////////
// JZS2
//
export function rJZS2Layer(arrayType, x, weights, activation='tanh', innerActivation='sigmoid') {
  let { W_xz, W_hz, b_z, W_hr, b_r, W_xh, W_hh, b_h, Pmat } = weights;

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_x_proj = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp2 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let z_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let r_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_tm1 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    mvprod(temp_hz, W_hz.transpose(1, 0), h_tm1);
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, temp_hz);
    ops.addeq(z_t, b_z);
    activationFuncs[innerActivation](z_t);

    mvprod(temp_x_proj, Pmat.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_x_proj);
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

    ops.mul(temp, z_t, h_t);
    ops.assign(temp2, z_t);
    ops.addseq(ops.negeq(temp2), 1);
    ops.muleq(temp2, h_tm1);
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


///////////////////////////////////////////////////////
// JZS3
//
export function rJZS3Layer(arrayType, x, weights, activation='tanh', innerActivation='sigmoid') {
  let { W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h } = weights;

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp2 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let z_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hz = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let r_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hr = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_xh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_hh = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_tm1 = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let h_tm1_temp = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    ops.assign(h_tm1_temp, h_tm1);
    mvprod(temp_hz, W_hz.transpose(1, 0), activationFuncs['tanh'](h_tm1_temp));
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

    ops.mul(temp, z_t, h_t);
    ops.assign(temp2, z_t);
    ops.addseq(ops.negeq(temp2), 1);
    ops.muleq(temp2, h_tm1);
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
