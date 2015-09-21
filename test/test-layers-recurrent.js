import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import { rGRULayer } from '../src/layers/recurrent';

const EPSILON = almostEqual.DBL_EPSILON;

describe('Layer: recurrent', function() {
  let input = ndarray(new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), [2, 4]);

  describe('gated recurrent unit (GRU)', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_GRU.json');
      for (let key in weights) {
        let packed = pack(weights[key]);
        packed.data = ndarray(new Float64Array(packed.data), packed.);
        weights[key] = packed;
      }

      let y = rGRULayer(input, weights);
      let expected = new Float64Array([0.5854064873930955, 0.6566667408032925, 0.3494883836663248, 0.3578150449007096]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        for (let j = 0; j < y.shape[1]; j++) {
          assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
        }
      }
      done();
    });
  });
});
