import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from '../src/lib/ndarray-pack';
import { rLSTMLayer, rGRULayer } from '../src/layers/recurrent';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: recurrent', function() {
  let arrayType = Float64Array;
  let input = pack(arrayType, [[0.1, 0.0, 0.9, 0.6], [0.5, 0.5, 0.5, 0.3]]);

  describe('long short-term memory (LSTM) serialized from Keras', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_LSTM_keras.json');

      let y = rLSTMLayer(arrayType, input, weights);
      let expected = new Float64Array([0.15660709142684937, -0.12310830503702164, 0.3947620987892151, 0.4411243498325348]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });

    it('should output the correct hidden state at the last timestep (non-square weights)', (done) => {
      let weights = require('./fixtures/test_weights_nonsquare_LSTM_keras.json');

      let y = rLSTMLayer(arrayType, input, weights);
      let expected = new Float64Array([0.5871871870691755, -0.002644894317063672, -0.19540790878272982]);

      assert.deepEqual(y.shape, [3]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('gated recurrent unit (GRU) serialized from Keras', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_GRU_keras.json');

      let y = rGRULayer(arrayType, input, weights);
      let expected = new Float64Array([0.5854064873930955, 0.6566667408032925, 0.3494883836663248, 0.3578150449007096]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });

    it('should output the correct hidden state at the last timestep (non-square weights)', (done) => {
      let weights = require('./fixtures/test_weights_nonsquare_GRU_keras.json');

      let y = rGRULayer(arrayType, input, weights);
      let expected = new Float64Array([0.586827501766111, 0.6312318587724021, 0.6273311716550194]);

      assert.deepEqual(y.shape, [3]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });
  });

});
