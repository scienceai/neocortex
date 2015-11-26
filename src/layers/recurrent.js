import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from '../lib/ndarray-pack';
import mvprod from '../lib/cpu/matrix-vector-product';
import * as activationFuncs from '../functions/activations';

/*
* Recurrent neural network layers
*
* shape of input tensor: timesteps x dimensions
*/

///////////////////////////////////////////////////////
// LSTM
//
export function rLSTMLayer(arrayType, x, weights, activation='tanh', inner_activation='hard_sigmoid', return_sequences=false) {
  let W_xi = pack(arrayType, weights['W_xi']);
  let W_hi = pack(arrayType, weights['W_hi']);
  let b_i = pack(arrayType, weights['b_i']);
  let W_xc = pack(arrayType, weights['W_xc']);
  let W_hc = pack(arrayType, weights['W_hc']);
  let b_c = pack(arrayType, weights['b_c']);
  let W_xf = pack(arrayType, weights['W_xf']);
  let W_hf = pack(arrayType, weights['W_hf']);
  let b_f = pack(arrayType, weights['b_f']);
  let W_xo = pack(arrayType, weights['W_xo']);
  let W_ho = pack(arrayType, weights['W_ho']);
  let b_o = pack(arrayType, weights['b_o']);

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);

  let i_t = ndarray(new arrayType(b_i.shape[0]), [b_i.shape[0]]);
  let temp_xi = ndarray(new arrayType(b_i.shape[0]), [b_i.shape[0]]);
  let temp_hi = ndarray(new arrayType(b_i.shape[0]), [b_i.shape[0]]);
  let f_t = ndarray(new arrayType(b_f.shape[0]), [b_f.shape[0]]);
  let temp_xf = ndarray(new arrayType(b_f.shape[0]), [b_f.shape[0]]);
  let temp_hf = ndarray(new arrayType(b_f.shape[0]), [b_f.shape[0]]);
  let o_t = ndarray(new arrayType(b_o.shape[0]), [b_o.shape[0]]);
  let temp_xo = ndarray(new arrayType(b_o.shape[0]), [b_o.shape[0]]);
  let temp_ho = ndarray(new arrayType(b_o.shape[0]), [b_o.shape[0]]);

  let c_t = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);
  let temp_xc = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);
  let temp_hc = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);
  let temp_fc = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);
  let c_tm1 = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);

  let h_t = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);
  let h_tm1 = ndarray(new arrayType(b_c.shape[0]), [b_c.shape[0]]);

  let h_seq;
  if (return_sequences) {
    h_seq = ndarray(new arrayType(x.shape[0] * b_c.shape[0]), [x.shape[0], b_c.shape[0]]);
  }

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xi, W_xi.transpose(1, 0), x_t);
    mvprod(temp_hi, W_hi.transpose(1, 0), h_tm1);
    ops.assigns(i_t, 0);
    ops.addeq(i_t, temp_xi);
    ops.addeq(i_t, temp_hi);
    ops.addeq(i_t, b_i);
    activationFuncs[inner_activation](i_t);

    mvprod(temp_xf, W_xf.transpose(1, 0), x_t);
    mvprod(temp_hf, W_hf.transpose(1, 0), h_tm1);
    ops.assigns(f_t, 0);
    ops.addeq(f_t, temp_xf);
    ops.addeq(f_t, temp_hf);
    ops.addeq(f_t, b_f);
    activationFuncs[inner_activation](f_t);

    mvprod(temp_xo, W_xo.transpose(1, 0), x_t);
    mvprod(temp_ho, W_ho.transpose(1, 0), h_tm1);
    ops.assigns(o_t, 0);
    ops.addeq(o_t, temp_xo);
    ops.addeq(o_t, temp_ho);
    ops.addeq(o_t, b_o);
    activationFuncs[inner_activation](o_t);

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

    if (return_sequences) {
      ops.assign(h_seq.pick(i, null), h_t);
    }
  }

  return h_t;
}


///////////////////////////////////////////////////////
// GRU
//
export function rGRULayer(arrayType, x, weights, activation='tanh', inner_activation='hard_sigmoid', return_sequences=false) {
  let W_xz = pack(arrayType, weights['W_xz']);
  let W_hz = pack(arrayType, weights['W_hz']);
  let b_z = pack(arrayType, weights['b_z']);
  let W_xr = pack(arrayType, weights['W_xr']);
  let W_hr = pack(arrayType, weights['W_hr']);
  let b_r = pack(arrayType, weights['b_r']);
  let W_xh = pack(arrayType, weights['W_xh']);
  let W_hh = pack(arrayType, weights['W_hh']);
  let b_h = pack(arrayType, weights['b_h']);

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp2 = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let z_t = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_xz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_hz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let r_t = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_xr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_hr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let h_t = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_xh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_hh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let h_tm1 = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);

  let h_seq;
  if (return_sequences) {
    h_seq = ndarray(new arrayType(x.shape[0] * b_h.shape[0]), [x.shape[0], b_h.shape[0]]);
  }

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    mvprod(temp_hz, W_hz.transpose(1, 0), h_tm1);
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, temp_hz);
    ops.addeq(z_t, b_z);
    activationFuncs[inner_activation](z_t);

    mvprod(temp_xr, W_xr.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_xr);
    ops.addeq(r_t, temp_hr);
    ops.addeq(r_t, b_r);
    activationFuncs[inner_activation](r_t);

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

    if (return_sequences) {
      ops.assign(h_seq.pick(i, null), h_t);
    }
  }

  return h_t;
}


///////////////////////////////////////////////////////
// JZS1
//
export function rJZS1Layer(arrayType, x, weights, activation='tanh', inner_activation='sigmoid', return_sequences=false) {
  let W_xz = pack(arrayType, weights['W_xz']);
  let b_z = pack(arrayType, weights['b_z']);
  let W_xr = pack(arrayType, weights['W_xr']);
  let W_hr = pack(arrayType, weights['W_hr']);
  let b_r = pack(arrayType, weights['b_r']);
  let W_hh = pack(arrayType, weights['W_hh']);
  let b_h = pack(arrayType, weights['b_h']);
  let Pmat = pack(arrayType, weights['Pmat']);

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp2 = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let z_t = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_xz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let r_t = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_xr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_hr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let h_t = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_x_proj = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_hh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let h_tm1 = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);

  let h_seq;
  if (return_sequences) {
    h_seq = ndarray(new arrayType(x.shape[0] * b_h.shape[0]), [x.shape[0], b_h.shape[0]]);
  }

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, b_z);
    activationFuncs[inner_activation](z_t);

    mvprod(temp_xr, W_xr.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_xr);
    ops.addeq(r_t, temp_hr);
    ops.addeq(r_t, b_r);
    activationFuncs[inner_activation](r_t);

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

    if (return_sequences) {
      ops.assign(h_seq.pick(i, null), h_t);
    }
  }

  return h_t;
}


///////////////////////////////////////////////////////
// JZS2
//
export function rJZS2Layer(arrayType, x, weights, activation='tanh', inner_activation='sigmoid', return_sequences=false) {
  let W_xz = pack(arrayType, weights['W_xz']);
  let W_hz = pack(arrayType, weights['W_hz']);
  let b_z = pack(arrayType, weights['b_z']);
  let W_hr = pack(arrayType, weights['W_hr']);
  let b_r = pack(arrayType, weights['b_r']);
  let W_xh = pack(arrayType, weights['W_xh']);
  let W_hh = pack(arrayType, weights['W_hh']);
  let b_h = pack(arrayType, weights['b_h']);
  let Pmat  = pack(arrayType, weights['Pmat']);

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp_x_proj = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp2 = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let z_t = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_xz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_hz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let r_t = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_hr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let h_t = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_xh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_hh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let h_tm1 = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);

  let h_seq;
  if (return_sequences) {
    h_seq = ndarray(new arrayType(x.shape[0] * b_h.shape[0]), [x.shape[0], b_h.shape[0]]);
  }

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    mvprod(temp_hz, W_hz.transpose(1, 0), h_tm1);
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, temp_hz);
    ops.addeq(z_t, b_z);
    activationFuncs[inner_activation](z_t);

    mvprod(temp_x_proj, Pmat.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_x_proj);
    ops.addeq(r_t, temp_hr);
    ops.addeq(r_t, b_r);
    activationFuncs[inner_activation](r_t);

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

    if (return_sequences) {
      ops.assign(h_seq.pick(i, null), h_t);
    }
  }

  return h_t;
}


///////////////////////////////////////////////////////
// JZS3
//
export function rJZS3Layer(arrayType, x, weights, activation='tanh', inner_activation='sigmoid', return_sequences=false) {
  let W_xz = pack(arrayType, weights['W_xz']);
  let W_hz = pack(arrayType, weights['W_hz']);
  let b_z = pack(arrayType, weights['b_z']);
  let W_xr = pack(arrayType, weights['W_xr']);
  let W_hr = pack(arrayType, weights['W_hr']);
  let b_r = pack(arrayType, weights['b_r']);
  let W_xh = pack(arrayType, weights['W_xh']);
  let W_hh = pack(arrayType, weights['W_hh']);
  let b_h = pack(arrayType, weights['b_h']);

  let x_t = ndarray(new arrayType(x.shape[1]), [x.shape[1]]);
  let temp = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp2 = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let z_t = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_xz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let temp_hz = ndarray(new arrayType(b_z.shape[0]), [b_z.shape[0]]);
  let r_t = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_xr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let temp_hr = ndarray(new arrayType(b_r.shape[0]), [b_r.shape[0]]);
  let h_t = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_xh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let temp_hh = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let h_tm1 = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);
  let h_tm1_temp = ndarray(new arrayType(b_h.shape[0]), [b_h.shape[0]]);

  let h_seq;
  if (return_sequences) {
    h_seq = ndarray(new arrayType(x.shape[0] * b_h.shape[0]), [x.shape[0], b_h.shape[0]]);
  }

  function _step() {
    ops.assign(h_tm1, h_t);

    mvprod(temp_xz, W_xz.transpose(1, 0), x_t);
    ops.assign(h_tm1_temp, h_tm1);
    mvprod(temp_hz, W_hz.transpose(1, 0), activationFuncs['tanh'](h_tm1_temp));
    ops.assigns(z_t, 0);
    ops.addeq(z_t, temp_xz);
    ops.addeq(z_t, temp_hz);
    ops.addeq(z_t, b_z);
    activationFuncs[inner_activation](z_t);

    mvprod(temp_xr, W_xr.transpose(1, 0), x_t);
    mvprod(temp_hr, W_hr.transpose(1, 0), h_tm1);
    ops.assigns(r_t, 0);
    ops.addeq(r_t, temp_xr);
    ops.addeq(r_t, temp_hr);
    ops.addeq(r_t, b_r);
    activationFuncs[inner_activation](r_t);

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

    if (return_sequences) {
      ops.assign(h_seq.pick(i, null), h_t);
    }
  }

  return h_t;
}
