import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import { dropoutLayer } from '../src/layers/dropout';

describe('Layer: dropout', function() {
  let x = ndarray(new Float64Array([0.5, 0.6, 1.2]), [3]);

  it('should produce expected', (done) => {
    let expected = [0.25, 0.3, 0.6];
    let y = dropoutLayer(Float64Array, x, 0.5);
    assert.deepEqual(y.shape, [3]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], almostEqual.FLT_EPSILON, almostEqual.FLT_EPSILON));
    }
    done();
  });

  it('should produce expected', (done) => {
    let expected = [0.45, 0.54, 1.08];
    let y = dropoutLayer(Float64Array, x, 0.1);
    assert.deepEqual(y.shape, [3]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], almostEqual.FLT_EPSILON, almostEqual.FLT_EPSILON));
    }
    done();
  });
});
