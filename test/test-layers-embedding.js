import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { embeddingLayer } from '../src/layers/embedding';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: embedding', function() {
  let input = ndarray(new Int32Array([0,0,0,1,2,3]), [6]);
  let E = [[0, 0, 0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];

  it('should create zero-masked embedding matrix', (done) => {
    let y = embeddingLayer(Float64Array, input, { E }, true);
    assert.deepEqual(y.shape, [3, 3]);
    for (let i = 0; i < y.shape[0]; i++) {
      for (let j = 0; j < y.shape[1]; j++) {
        assert(almostEqual(y.get(i, j), E[i+1][j], EPSILON, EPSILON));
      }
    }
    done();
  });

  it('should create non-zero-masked embedding matrix', (done) => {
    let y = embeddingLayer(Float64Array, input, { E }, false);
    assert.deepEqual(y.shape, [6, 3]);
    for (let i = 0; i < y.shape[0]; i++) {
      for (let j = 0; j < y.shape[1]; j++) {
        assert(almostEqual(y.get(i, j), E[input.get(i)][j], EPSILON, EPSILON));
      }
    }
    done();
  });
});
