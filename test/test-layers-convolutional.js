import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import { convolution2DLayer } from '../src/layers/convolutional';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: convolutional', function() {
  let input = pack([
    [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],
    [[16,17,18,19],[20,21,22,23],[24,25,26,27],[28,29,30,31]]
  ]);
  let arrayType = Float64Array;

  describe('2D convolutional layer serialized from Keras', function() {
    it('should output the correct tensor [border mode: valid]', (done) => {
      let weights = require('./fixtures/test_weights_convolution2d_keras.json');

      let y = convolution2DLayer(arrayType, input, weights,
        nb_filter=5, stack_size=2, nb_row=4, nb_col=3,
        border_mode='valid', subsample=[1,1], activation='linear');

      let expected = pack([[[-23.931154251098633, -25.2711181640625]], [[3.360344171524048, 3.208177328109741]], [[2.332014322280884, 2.3287289142608643]], [[-26.280771255493164, -27.71440887451172]], [[54.16725158691406, 56.917640686035156]]]);

      assert.deepEqual(y.shape, [5,1,2]);
      for (let i = 0; i < y.shape[0]; i++) {
        for (let j = 0; j < y.shape[1]; j++) {
          for (let k = 0; k < y.shape[2]; k++) {
            assert(almostEqual(y.get(i,j,k), expected.get(i,j,k), EPSILON, EPSILON));
          }
        }
      }
      done();
    });
  });
});
