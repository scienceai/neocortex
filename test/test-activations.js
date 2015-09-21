import assert from 'assert';
import ndarray from 'ndarray';
import almostEqual from 'almost-equal';
import { linear, relu, sigmoid, sigmoidHard, tanh, softmax } from '../src/functions/activations';

const EPSILON = almostEqual.DBL_EPSILON;

describe('Activation functions [baseline]', function () {
  let x = [ 0.01, 0.03, -0.01, 0.05, 0 ];
  let repeat = 17;
  while(repeat--) {
    x = x.concat(x);
  }

  let x_float64;
  // activation functions mutate data in-place, thus we need new vectors for each test
  beforeEach((done) => {
    x_float64 = ndarray(new Float64Array(x), [Math.pow(2,17)*5]);
    done();
  });

  describe('linear', function () {
    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float64 = linear(x_float64);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float64.data.every((y_i, i) => Math.abs(y_i - x_float64.data[i]) < EPSILON));
      done();
    });
  });

  describe('relu', function () {
    let expected = [ 0.01, 0.03, 0.01, 0.05, 0];
    let repeat = 17;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    let expected_float64 = ndarray(new Float64Array(expected), [Math.pow(2,17)*5]);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float64 = relu(x_float64);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float64.data.every((y_i, i) => Math.abs(y_i - expected_float64.data[i]) < EPSILON));
      done();
    });
  });

  describe('sigmoid', function () {
    let expected = [ 0.50249998, 0.50749944, 0.49750002, 0.5124974, 0.5 ];
    let repeat = 17;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    let expected_float64 = ndarray(new Float64Array(expected), [Math.pow(2,17)*5]);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float64 = sigmoid(x_float64);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float64.data.every((y_i, i) => Math.abs(y_i - expected_float64.data[i]) < EPSILON));
      done();
    });
  });

  describe('sigmoidHard', function () {
    let expectedSigmoid = [ 0.50249998, 0.50749944, 0.49750002, 0.5124974, 0.5 ];
    let repeat = 17;
    while(repeat--) {
      expectedSigmoid = expectedSigmoid.concat(expectedSigmoid);
    }
    let expected_float64 = ndarray(new Float64Array(expectedSigmoid), [Math.pow(2,17)*5]);

    it('should be pretty close to normal sigmoid', (done) => {
      let start = new Date().getTime();
      let y_float64 = sigmoidHard(x_float64);
      console.log(`      ${new Date().getTime() - start} ms`);
      // should be pretty close to normal sigmoid
      assert(y_float64.data.every((y_i, i) => Math.abs(y_i - expected_float64.data[i]) < 0.01));
      done();
    });
  });

  describe('tanh', function () {
    let expected = [ 0.00999967, 0.029991, -0.00999967, 0.04995837, 0 ];
    let repeat = 17;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    let expected_float64 = ndarray(new Float64Array(expected), [Math.pow(2,17)*5]);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float64 = tanh(x_float64);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float64.data.every((y_i, i) => Math.abs(y_i - expected_float64.data[i]) < EPSILON));
      done();
    });
  });

  describe('softmax', function () {
    let expected = [ 0.19875734, 0.20277251, 0.19482169, 0.20686879, 0.19677968 ];
    let repeat = 17;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    expected = expected.map(z => 5 * z / expected.length);
    let expected_float64 = ndarray(new Float64Array(expected), [Math.pow(2,17)*5]);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float64 = softmax(x_float64);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float64.data.every((y_i, i) => Math.abs(y_i - expected_float64.data[i]) < EPSILON));
      done();
    });
  });

});
