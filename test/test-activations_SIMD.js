require('./simd-shim');

import assert from 'assert';
import { linear, relu, sigmoid, sigmoidHard, tanh, softmax } from '../src/functions/activations_SIMD';

const EPSILON = 0.00001;

describe('Activation functions [SIMD]', function () {
  let x = [ 0.01, 0.03, -0.01, 0.05, 0 ];
  let repeat = 17;
  while(repeat--) {
    x = x.concat(x);
  }
  let x_float32 = new Float32Array(x);

  describe('linear', function () {
    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y = linear(x);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert.equal(y, x);
      done();
    });
  });

  describe('relu', function () {
    let expected = [ 0.01, 0.03, 0, 0.05, 0];
    let repeat = 17;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    let expected_float32 = new Float32Array(expected);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float32 = relu(x_float32);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float32.every((y_i, i) => Math.abs(y_i - expected_float32[i]) < EPSILON));
      done();
    });
  });

  describe('sigmoid', function () {
    let expected = [ 0.50249998, 0.50749944, 0.49750002, 0.5124974, 0.5 ];
    let repeat = 17;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    let expected_float32 = new Float32Array(expected);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y_float32 = sigmoid(x_float32);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float32.every((y_i, i) => Math.abs(y_i - expected_float32[i]) < 0.01));
      done();
    });
  });

  describe('sigmoidHard', function () {
    let expectedSigmoid = [ 0.50249998, 0.50749944, 0.49750002, 0.5124974, 0.5 ];
    let repeat = 17;
    while(repeat--) {
      expectedSigmoid = expectedSigmoid.concat(expectedSigmoid);
    }
    let expectedSigmoid_float32 = new Float32Array(expectedSigmoid);

    it('should be pretty close to normal sigmoid', (done) => {
      let start = new Date().getTime();
      let y_float32 = sigmoidHard(x_float32);
      console.log(`      ${new Date().getTime() - start} ms`);
      // should be pretty close to normal sigmoid
      assert(y_float32.every((y_i, i) => Math.abs(y_i - expectedSigmoid_float32[i]) < 0.01));
      done();
    });
  });

});
