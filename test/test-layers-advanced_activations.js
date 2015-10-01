import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from 'ndarray-pack';
import { leakyReLULayer, parametricReLULayer, parametricSoftplusLayer, thresholdedLinearLayer, thresholdedReLuLayer } from '../src/layers/advanced_activations';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: advanced activations', function() {
  let input = pack([1,0,-0.6,0.9,2,-1.2,0.4,-3]);
  let arrayType = Float64Array;

  describe('leaky ReLU (alpha = 0.3)', function() {
    it('should output the correct values', (done) => {

      let y = leakyReLULayer(arrayType, input, 0.3);
      let expected = pack([1.0, 0.0, -0.18000000715255737, 0.8999999761581421, 2.0, -0.36000001430511475, 0.4000000059604645, -0.9000000357627869]);

      assert.deepEqual(y.shape, input.shape);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('parametric ReLU', function() {
    it('should output the correct values', (done) => {
      let weights = {alphas: [0.1,0.02,0.4,0.2,0.5,0.9,1.0,0.0]};

      let y = parametricReLULayer(arrayType, input, weights);
      let expected = pack([1.0, 0.0, -0.24000000953674316, 0.8999999761581421, 2.0, -1.0800000429153442, 0.4000000059604645, 0.0]);

      assert.deepEqual(y.shape, input.shape);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('parametric softplus', function() {
    it('should output the correct values', (done) => {
      let weights = {alphas: [0.1,0.02,0.4,0.2,0.5,0.9,1.0,0.0], betas: [0,1,2,3,4,5,6,7]};

      let y = parametricSoftplusLayer(arrayType, input, weights);
      let expected = pack([0.06931471824645996, 0.013862943276762962, 0.10531298071146011, 0.5530087351799011, 4.0001678466796875, 0.0022281166166067123, 2.4868361949920654, 0.0]);

      assert.deepEqual(y.shape, input.shape);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('thresholded linear (theta = 1.0)', function() {
    it('should output the correct values', (done) => {

      let y = thresholdedLinearLayer(arrayType, input, 1.0);
      let expected = pack([1.0, 0.0, 0.0, 0.0, 2.0, -1.2000000476837158, 0.0, -3.0]);

      assert.deepEqual(y.shape, input.shape);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
      }
      done();
    });
  });

  describe('thresholded ReLU (theta = 1.0)', function() {
    it('should output the correct values', (done) => {

      let y = thresholdedReLuLayer(arrayType, input, 1.0);
      let expected = pack([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);

      assert.deepEqual(y.shape, input.shape);
      for (let i = 0; i < y.shape[0]; i++) {
        assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
      }
      done();
    });
  });
});
