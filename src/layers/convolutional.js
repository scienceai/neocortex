import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import pack from 'ndarray-pack';
import convolve from 'ndarray-convolve';
import mvprod from '../lib/matrix-vector-product.js';
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
    ops.assign(x_mod.hi(x.shape[1] + nb_row - 1, x.shape[2] + nb_col - 1).lo(nb_row-1, nb_col-1), x);
  }

  for (let filter = 0; filter < nb_filter; filter++) {

    let convTempSum = ndarray(new arrayType(rows_new * cols_new), [rows_new, cols_new]);
    for (let stack = 0; stack < stack_size; stack++) {

      let convTemp = ndarray(new arrayType(rows_new * cols_new), [rows_new, cols_new]);

      if (border_mode === 'valid') {

        convolve(convTemp, x.pick(stack, null, null), W.pick(filter, stack, null, null));

      } else if (border_mode === 'same') {

        let convTempFull = ndarray(new arrayType((x.shape[1] + nb_row - 1) * (x.shape[2] + nb_col - 1)), [x.shape[1] + nb_row - 1, x.shape[2] + nb_col - 1]);
        convolve(convTempFull, x_mod.pick(stack, null, null), W.pick(filter, stack, null, null));
        let shift_x = Math.floor((nb_row - 1) / 2);
        let shift_y = Math.floor((nb_col - 1) / 2);
        ops.assign(convTemp, convTempFull.hi(x.shape[1] + shift_x, x.shape[2] + shift_y).lo(shift_x, shift_y));

      } else if (border_mode === 'full') {

        convolve(convTemp, x_mod.pick(stack, null, null), W.pick(filter, stack, null, null));

      }

      ops.sumeq(convTempSum, convTemp);
    }

    ops.addseq(convTempSum, b.get(filter));
    ops.assign(y.pick(filter, null, null), convTempSum);
  }

  activationFuncs[activation](y);
  return y;
}
