import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import { batchNormalizationLayer } from '../src/layers/normalization';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: normalization', function() {
  let input = pack([1,0,-0.6,0.9,2,-1.2,0.4,-3]);
  let arrayType = Float64Array;

  describe('batch normalization', function() {
    it('should output the correct values', (done) => {
      let weights = {
        gamma: [0.042860303074121475, 0.03142599016427994, -0.022097120061516762, 0.0349823459982872, -0.009010647423565388, -0.000799372442997992, -0.02932022511959076, 0.003696359694004059],
        beta: [0, 1, 2, 3, 4, 5, 6, 7],
        mean: [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4],
        std: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      };

      let y = batchNormalizationLayer(arrayType, input, weights);
      let expected = pack([0.03857423737645149, 1.0031425952911377, 2.0176777839660645, 3.038480520248413, 3.9846818447113037, 5.0007195472717285, 6.0, 6.990389347076416]);

      assert.deepEqual(y.shape, input.shape);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
      }
      done();
    });
  });
});
