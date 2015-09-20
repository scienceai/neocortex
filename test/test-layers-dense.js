import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import { denseLayer } from '../src/layers/dense';

describe('Layer: dense', function() {
  let x = ndarray(new Float32Array([0.25, 0.5, 0.75]), [3]);
  let W = ndarray(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), [3, 3]);
  let b = ndarray(new Float32Array([1, 2, 3]), [3]);

  it('should produce expected W*x + b', (done) => {
    let expected = [1.35, 2.8, 4.25];
    let y = denseLayer(x, W, b);
    assert.deepEqual(y.shape, [3]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], almostEqual.FLT_EPSILON, almostEqual.FLT_EPSILON));
    }
    done();
  });

  it('should produce expected softmax(W*x + b)', (done) => {
    let expected = [0.0426671, 0.18189475, 0.77543815];
    let y = denseLayer(x, W, b, 'softmax');
    assert.deepEqual(y.shape, [3]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], almostEqual.FLT_EPSILON, almostEqual.FLT_EPSILON));
    }
    done();
  });
});
