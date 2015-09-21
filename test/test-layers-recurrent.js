import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import { rGRULayer } from '../src/layers/recurrent';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: recurrent', function() {
  let input = pack([[0.1, 0.0, 0.9, 0.6], [0.5, 0.5, 0.5, 0.3]]);

  describe('gated recurrent unit (GRU) serialized from Keras', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_GRU_keras.json');
      for (let key in weights) {
        // pack creates Float64Array ndarrays
        // TODO: need to convert to Float32Array if set as default
        weights[key] = pack(weights[key]);
      }

      let y = rGRULayer(input, weights);
      let expected = new Float64Array([0.5854064873930955, 0.6566667408032925, 0.3494883836663248, 0.3578150449007096]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });
  });
});