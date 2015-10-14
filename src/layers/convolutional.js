import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import convolve from '../lib/cpu/convolve-2d';
import mvprod from '../lib/cpu/matrix-vector-product';
import * as activationFuncs from '../functions/activations';

/*
* Convolutional neural network layers
*
* shape of input tensor: timesteps x dimensions
* 2D: timesteps x rows x cols
* 1D: timesteps x rows
*/

///////////////////////////////////////////////////////
// 2D convolutional layer
//
// TODO:
// - subsample
export function convolution2DLayer(arrayType, x, weights,
  nb_filter=64, nb_row=3, nb_col=3,
  border_mode='valid',
  subsample=[1,1],
  activation='relu') {

  let stack_size = x.shape[0];

  let W = pack(weights['W']);
  let b = pack(weights['b']);

  let stacks_new = nb_filter;
  let rows_new = x.shape[1];
  let cols_new = x.shape[2];
  if (border_mode === 'valid') {
    rows_new = x.shape[1] - nb_row + 1;
    cols_new = x.shape[2] - nb_col + 1;
  } else if (border_mode === 'same') {
    rows_new = x.shape[1];
    cols_new = x.shape[2];
  } else if (border_mode === 'full') {
    rows_new = x.shape[1] + nb_row - 1;
    cols_new = x.shape[2] + nb_col - 1;
  }

  let y = ndarray(new arrayType(nb_filter * rows_new * cols_new), [nb_filter, rows_new, cols_new]);

  let x_mod;
  if (border_mode === 'same' || border_mode === 'full') {
    // zero-padding
    x_mod = ndarray(new arrayType(x.shape[0] * (x.shape[1] + 2*(nb_row-1)) * (x.shape[2] + 2*(nb_col-1))), [x.shape[0], x.shape[1] + 2*(nb_row-1), x.shape[2] + 2*(nb_col-1)]);

    ops.assign(x_mod.hi(stack_size, x.shape[1] + nb_row - 1, x.shape[2] + nb_col - 1).lo(0, nb_row - 1, nb_col - 1), x);
  } else if (border_mode === 'valid') {
    x_mod = ndarray(new arrayType(x.data), x.shape);
  }

  // broadcast x
  let x_broadcasted = ndarray(new arrayType(nb_filter * x_mod.size), [nb_filter, x_mod.shape[0], x_mod.shape[1], x_mod.shape[2]]);
  for (let filter = 0; filter < nb_filter; filter++) {
    ops.assign(x_broadcasted.pick(filter, null, null, null), x_mod);
  }

  if (border_mode === 'valid') {

    convolve(y, x_broadcasted, W);

  } else if (border_mode === 'same') {

    let convTemp = ndarray(new arrayType(nb_filter * (x.shape[1] + nb_row - 1) * (x.shape[2] + nb_col - 1)), [nb_filter, x.shape[1] + nb_row - 1, x.shape[2] + nb_col - 1]);
    convolve(convTemp, x_broadcasted, W);
    let shift_x = Math.floor((nb_row - 1) / 2);
    let shift_y = Math.floor((nb_col - 1) / 2);

    ops.assign(y, convTemp.hi(nb_filter, rows_new + shift_x, cols_new + shift_y).lo(0, shift_x, shift_y));

  } else if (border_mode === 'full') {

    convolve(y, x_broadcasted, W);

  }

  // add bias
  for (let filter = 0; filter < nb_filter; filter++) {
    ops.addseq(y.pick(filter, null, null), b.get(filter));
  }

  activationFuncs[activation](y);

  return y;
}


///////////////////////////////////////////////////////
// 2D max-pooling layer
//
export function maxPooling2DLayer(arrayType, x, pool_size=[2,2], stride=null, ignore_border=true) {
  let rows_new = Math.floor(x.shape[1] / pool_size[0]);
  let cols_new = Math.floor(x.shape[2] / pool_size[1]);

  let y = ndarray(new arrayType(x.shape[0] * rows_new * cols_new), [x.shape[0], rows_new, cols_new]);

  // x.shape[0] wil represent the stack size
  for (let stack = 0; stack < x.shape[0]; stack++) {
    for (let i = 0; i < rows_new; i++) {
      for (let j = 0; j < cols_new; j++) {
        y.set(stack, i, j, ops.sup(x.pick(stack, null, null).hi((i+1)*pool_size[0], (j+1)*pool_size[1]).lo(i*pool_size[0], j*pool_size[1])));
      }
    }
  }

  return y;
}

///////////////////////////////////////////////////////
// 1D convolutional layer
//
// note: convolution performed over time dimension
//
// TODO:
// - subsample
export function convolution1DLayer(arrayType, x, weights,
  nb_filter=64, filter_length=3,
  border_mode='valid',
  subsample_length=1,
  activation='relu') {

  let W = pack(weights['W']);
  let b = pack(weights['b']);

  let x_mod = ndarray(new arrayType(x.size), [x.shape[1], x.shape[0], 1]);
  ops.assign(x_mod.pick(null, null, 0), x.transpose(1, 0));

  let y_mod = convolution2DLayer(arrayType, x_mod, weights,
    nb_filter, filter_length, 1,
    border_mode,
    [subsample_length, 1],
    activation);

  let y = ndarray(new arrayType(y_mod.size), [y_mod.shape[1], y_mod.shape[0]]);
  ops.assign(y, y_mod.pick(null, null, 0).transpose(1, 0));

  return y;
}


///////////////////////////////////////////////////////
// 1D max-pooling layer
//
export function maxPooling1DLayer(arrayType, x, pool_length=2, stride=null, ignore_border=true) {
  let len_new = Math.floor(x.shape[0] / pool_length);

  let y = ndarray(new arrayType(len_new * x.shape[1]), [len_new, x.shape[1]]);

  for (let i = 0; i < len_new; i++) {
    for (let j = 0; j < x.shape[1]; j++) {
      y.set(i, j, ops.sup(x.pick(null, j).hi((i+1)*pool_length).lo(i*pool_length)));
    }
  }

  return y;
}
