import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from '../src/lib/ndarray-pack';
import { flattenLayer } from '../src/layers/flatten';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: flatten', function() {
  let arrayType = Float64Array;

  it('should flatten 1-dimensional tensor', (done) => {
    let input = pack(arrayType, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let expected = pack(arrayType, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let y = flattenLayer(arrayType, input);
    assert.deepEqual(y.shape, [12]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
    }
    done();
  });

  it('should flatten 2-dimensional tensor', (done) => {
    let input = pack(arrayType, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]);
    let expected = pack(arrayType, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let y = flattenLayer(arrayType, input);
    assert.deepEqual(y.shape, [12]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
    }
    done();
  });

  it('should flatten 3-dimensional tensor', (done) => {
    let input = pack(arrayType, [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);
    let expected = pack(arrayType, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let y = flattenLayer(arrayType, input);
    assert.deepEqual(y.shape, [12]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected.get(i), EPSILON, EPSILON));
    }
    done();
  });
});
