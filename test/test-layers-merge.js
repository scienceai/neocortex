import assert from 'assert';
import almostEqual from 'almost-equal';
import ndarray from 'ndarray';
import pack from '../src/lib/ndarray-pack';
import { mergeLayer } from '../src/layers/merge';

const EPSILON = almostEqual.FLT_EPSILON;

describe('Layer: merge', function() {
  let arrayType = Float64Array;

  let input = [ndarray(new Float64Array([0.25, 0.5, 0.75]), [3]), ndarray(new Float64Array([0.65, 0.35, 0.15]), [3])];
  let branches = [
    [{
      'layerName': 'denseLayer',
      'parameters': [{'W': [[0.3736618, 0.5745131],[1.0225855, 0.9838101],[0.7040277, -0.0223534]],'b': [0.0, 0.0]},'linear']
    }],
    [{
      'layerName': 'denseLayer',
      'parameters': [{'W': [[-1.0081574, 0.7686076],[1.0467454, -0.2916093],[0.2885694, -0.5222564]],'b': [0.0, 0.0]},'linear']
    }]
  ];

  it('should produce expected in `sum` mode', done => {
    let expected = [0.88707298, 0.93796146];
    let y = mergeLayer(arrayType, input, branches, 'sum');
    assert.deepEqual(y.shape, [2]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
    }
    done();
  });

  it('should produce expected in `ave` mode', done => {
    let expected = [0.44353649, 0.46898073];
    let y = mergeLayer(arrayType, input, branches, 'ave');
    assert.deepEqual(y.shape, [2]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
    }
    done();
  });

  it('should produce expected in `mul` mode', done => {
    let expected = [-0.27826163, 0.19750662];
    let y = mergeLayer(arrayType, input, branches, 'mul');
    assert.deepEqual(y.shape, [2]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
    }
    done();
  });

  it('should produce expected in `concat` mode', done => {
    let expected = [1.132728975, 0.6187682750000001, -0.24565601000000006, 0.31919322499999997];
    let y = mergeLayer(arrayType, input, branches, 'concat', -1);
    assert.deepEqual(y.shape, [4]);
    for (let i = 0; i < y.shape[0]; i++) {
      assert(almostEqual(y.get(i), expected[i], EPSILON, EPSILON));
    }
    done();
  });
});
