import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import { embeddingLayer } from '../src/layers/embedding';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: embedding', function() {
  let input = ndarray(new Int32Array([0,0,0,1,2,3]), [6]);
  let E = ndarray(new Float64Array(4*3), [4, 3]);
  ops.assign(E.pick(1, null), ndarray(new Float64Array([0.1, 0.2, 0.3]), [3]));
  ops.assign(E.pick(2, null), ndarray(new Float64Array([0.4, 0.5, 0.6]), [3]));
  ops.assign(E.pick(3, null), ndarray(new Float64Array([0.7, 0.8, 0.9]), [3]));

  it('should create zero-masked embedding matrix', (done) => {
    let y = embeddingLayer(Float64Array, input, { E });
    assert.deepEqual(y.shape, [3, 3]);
    for (let i = 0; i < y.shape[0]; i++) {
      for (let j = 0; j < y.shape[1]; j++) {
        assert(almostEqual(y.get(i, j), E.get(i+1, j), EPSILON, EPSILON));
      }
    }
    done();
  });
});
