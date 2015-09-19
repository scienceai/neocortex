require('./common');

import assert from 'assert';
import { linear, relu, sigmoid, sigmoidHard, tanh, softmax } from '../src/functions/activations';

const EPSILON = 0.00001;

describe('Activation functions', function () {
  let x = [ 0.01, 0.03, -0.01, 0.05, 0 ];
  let repeat = 18;
  while(repeat--) {
    x = x.concat(x);
  }
  let x_float32 = new Float32Array(x);

  describe('linear', function () {
    it('[baseline] should return expected result', (done) => {
      let start = new Date().getTime();
      let y = linear(x);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert.equal(y, x);
      done();
    });
  });

  describe('relu', function () {
    let expected = [ 0.01, 0.03, 0, 0.05, 0];
    let repeat = 18;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    let expected_float32 = new Float32Array(expected);

    it('[baseline] should return expected result', (done) => {
      let useSIMD = false;
      let start = new Date().getTime();
      let y = relu(x, useSIMD);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y.every((y_i, i) => Math.abs(y_i - expected[i]) < EPSILON));
      done();
    });

    it('[SIMD] should return expected result', (done) => {
      let useSIMD = true;
      let start = new Date().getTime();
      let y_float32 = relu(x_float32, useSIMD);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y_float32.every((y_i, i) => Math.abs(y_i - expected_float32[i]) < EPSILON));
      done();
    });
  });

  describe('sigmoid', function () {
    let expected = [ 0.50249998, 0.50749944, 0.49750002, 0.5124974, 0.5 ];
    let repeat = 18;
    while(repeat--) {
      expected = expected.concat(expected);
    }

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y = sigmoid(x);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y.every((y_i, i) => Math.abs(y_i - expected[i]) < EPSILON));
      done();
    });
  });

  describe('sigmoidHard', function () {
    let expectedSigmoid = [ 0.50249998, 0.50749944, 0.49750002, 0.5124974, 0.5 ];
    let repeat = 18;
    while(repeat--) {
      expectedSigmoid = expectedSigmoid.concat(expectedSigmoid);
    }

    it('should be pretty close to normal sigmoid', (done) => {
      let start = new Date().getTime();
      let y = sigmoidHard(x);
      console.log(`      ${new Date().getTime() - start} ms`);
      // should be pretty close to normal sigmoid
      assert(y.every((y_i, i) => Math.abs(y_i - expectedSigmoid[i]) < 0.01));
      done();
    });
  });

  describe('tanh', function () {
    let expected = [ 0.00999967, 0.029991, -0.00999967, 0.04995837, 0 ];
    let repeat = 18;
    while(repeat--) {
      expected = expected.concat(expected);
    }

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y = tanh(x);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y.every((y_i, i) => Math.abs(y_i - expected[i]) < EPSILON));
      done();
    });
  });

  describe('softmax', function () {
    let expected = [ 0.19875734, 0.20277251, 0.19482169, 0.20686879, 0.19677968 ];
    let repeat = 18;
    while(repeat--) {
      expected = expected.concat(expected);
    }
    expected = expected.map(z => 5 * z / expected.length);

    it('should return expected result', (done) => {
      let start = new Date().getTime();
      let y = softmax(x);
      console.log(`      ${new Date().getTime() - start} ms`);
      assert(y.every((y_i, i) => Math.abs(y_i - expected[i]) < EPSILON));
      done();
    });
  });

});
