import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from '../src/lib/ndarray-pack';
import { rLSTMLayer, rGRULayer, rJZS1Layer, rJZS2Layer, rJZS3Layer } from '../src/layers/recurrent';

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

  describe('mutated recurrent network 1 (JZS1) serialized from Keras', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_JZS1_keras.json');

      let y = rJZS1Layer(arrayType, input, weights);
      let expected = new Float64Array([0.7362973093986511, -0.023125506937503815, 0.8724695444107056, 0.316345751285553]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });

    it('should output the correct hidden state at the last timestep (non-square weights)', (done) => {
      let weights = require('./fixtures/test_weights_nonsquare_JZS1_keras.json');

      let y = rJZS1Layer(arrayType, input, weights);
      let expected = new Float64Array([0.7789963211705515, 0.6363054133364847, 0.5116611144594283]);

      assert.deepEqual(y.shape, [3]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('mutated recurrent network 2 (JZS2) serialized from Keras', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_JZS2_keras.json');

      let y = rJZS2Layer(arrayType, input, weights);
      let expected = new Float64Array([0.4409944713115692, 0.6610596179962158, 0.22975823283195496, -0.12029259651899338]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });

    it('should output the correct hidden state at the last timestep (non-square weights)', (done) => {
      let weights = require('./fixtures/test_weights_nonsquare_JZS2_keras.json');

      let y = rJZS2Layer(arrayType, input, weights);
      let expected = new Float64Array([0.6960307678380586, 0.63672076273643, 0.005374565782340368]);

      assert.deepEqual(y.shape, [3]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('mutated recurrent network 3 (JZS3) serialized from Keras', function() {
    it('should output the correct hidden state at the last timestep', (done) => {
      let weights = require('./fixtures/test_weights_JZS3_keras.json');

      let y = rJZS3Layer(arrayType, input, weights);
      let expected = new Float64Array([0.8178967237472534, 0.1090848445892334, 0.5106683969497681, 0.5130321979522705]);

      assert.deepEqual(y.shape, [4]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });

    it('should output the correct hidden state at the last timestep (non-square weights)', (done) => {
      let weights = require('./fixtures/test_weights_nonsquare_JZS3_keras.json');

      let y = rJZS3Layer(arrayType, input, weights);
      let expected = new Float64Array([0.6544535354606844, 0.39989062885298476, 0.7527888042830722]);

      assert.deepEqual(y.shape, [3]);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
      }
      done();
    });
  });
});
