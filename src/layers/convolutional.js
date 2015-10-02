import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import convolve from 'ndarray-convolve';
import mvprod from '../lib/matrix-vector-product';
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

  let rows_new = 0;
  let cols_new = 0;
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

  let y = ndarray(new arrayType(W.shape[0] * rows_new * cols_new), [W.shape[0], rows_new, cols_new]);

  let x_mod;
  if (border_mode === 'same' || border_mode === 'full') {
    // zero-padding
    x_mod = ndarray(new arrayType(x.shape[0] * (x.shape[1] + 2*(nb_row-1)) * (x.shape[2] + 2*(nb_col-1))), [x.shape[0], x.shape[1] + 2*(nb_row-1), x.shape[2] + 2*(nb_col-1)]);
    for (let stack = 0; stack < stack_size; stack++) {
      ops.assign(x_mod.pick(stack, null, null).hi(x.shape[1] + nb_row - 1, x.shape[2] + nb_col - 1).lo(nb_row - 1, nb_col - 1), x.pick(stack, null, null));
    }
  }

  for (let filter = 0; filter < nb_filter; filter++) {

    let convTemp = ndarray(new arrayType(x.size), x.shape);
    let convTemp2 = ndarray(new arrayType(rows_new * cols_new), [rows_new, cols_new]);
    let filter_weights = W.pick(filter, null, null, null).step(-1, 1, 1);

    let row_begin = 0;
    let col_begin = 0;

    if (border_mode === 'valid') {

      convolve(convTemp, x, filter_weights);

      row_begin = 1;
      col_begin = 1;
      ops.assign(convTemp2, convTemp.pick(0, null, null).hi(row_begin + rows_new, col_begin + cols_new).lo(row_begin, col_begin));

    } else if (border_mode === 'same') {

      let convTempFull = ndarray(new arrayType((x.shape[1] + nb_row - 1) * (x.shape[2] + nb_col - 1)), [x.shape[1] + nb_row - 1, x.shape[2] + nb_col - 1]);
      convolve(convTempFull, x_mod.pick(stack, null, null), filter_weights);
      let shift_x = Math.floor((nb_row - 1) / 2);
      let shift_y = Math.floor((nb_col - 1) / 2);
      ops.assign(convTemp, convTempFull.hi(x.shape[1] + shift_x, x.shape[2] + shift_y).lo(shift_x, shift_y));

    } else if (border_mode === 'full') {

      convolve(convTemp, x_mod.pick(stack, null, null), filter_weights);

    }

    ops.addseq(convTemp2, b.get(filter));
    ops.assign(y.pick(filter, null, null), convTemp2);
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
