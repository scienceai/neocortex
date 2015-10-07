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
*/

///////////////////////////////////////////////////////
// 2D convolutional layer
//
// TODO:
// - subsample
export function convolution2DLayer(arrayType, x, weights,
  nb_filter=64, stack_size=3, nb_row=3, nb_col=3,
  border_mode='valid',
  subsample=[1,1],
  activation='relu') {

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
export function maxPooling2DLayer(arrayType, x, poolsize=[2,2], stride=null, ignore_border=true) {
  let rows_new = Math.floor(x.shape[1] / poolsize[0]);
  let cols_new = Math.floor(x.shape[2] / poolsize[1]);

  let y = ndarray(new arrayType(x.shape[0] * rows_new * cols_new), [x.shape[0], rows_new, cols_new]);

  // x.shape[0] wil represent the stack size
  for (let stack = 0; stack < x.shape[0]; stack++) {
    for (let i = 0; i < rows_new; i++) {
      for (let j = 0; j < cols_new; j++) {
        y.set(stack, i, j, ops.sup(x.pick(stack, null, null).hi((i+1)*poolsize[0], (j+1)*poolsize[1]).lo(i*poolsize[0], j*poolsize[1])));
      }
    }
  }

  return y;
}
