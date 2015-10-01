import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import { denseLayer } from '../src/layers/dense';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: dense', function() {
  let x = ndarray(new Float64Array([0.25, 0.5, 0.75]), [3]);
  let W = [[-0.9270800352096558, 0.18572671711444855, 0.39262476563453674], [-0.23333904147148132, -0.44041740894317627, -0.7116944789886475], [-0.6425269842147827, -0.9207804799079895, 0.4250005781650543]];
  let b = [1.0, 0.5, 0.009999999776482582];

  it('should produce expected W*x + b', (done) => {
    let expected = [0.16966521739959717, -0.3643624186515808, 0.07105938345193863];
    let y = denseLayer(Float64Array, x, { W, b });
    assert.deepEqual(y.shape, [3]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
    }
    done();
  });

  it('should produce expected softmax(W*x + b)', (done) => {
    let expected = [0.4012295603752136, 0.235216423869133, 0.3635540306568146];
    let y = denseLayer(Float64Array, x, { W, b }, 'softmax');
    assert.deepEqual(y.shape, [3]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
    }
    done();
  });
});
