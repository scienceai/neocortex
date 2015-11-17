import ndarray from 'ndarray';
import ops from 'ndarray-ops';

export function dropoutLayer(arrayType, x, p=0.5, rescale=false) {

  // If rescale, rescaling is done at test time,
  // whereas if false, it is assumed to have been done at training time.
  // In Keras, currently this is done during training time and rescale should be false
  if (rescale) {
    let y = ndarray(new arrayType(x.size), x.shape);
    ops.muls(y, x, 1 - p);
    return y;
  } else {
    return x;
  }

}
