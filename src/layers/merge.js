import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import * as layerFuncs from '../layers';

export function mergeLayer(arrayType, x_branches, branches, mode='concat', concat_axis=-1, dot_axes=-1) {
  // `x_branches` must be same nested-array shape as `branches`
  // `dot` mode not implemented
  if (x_branches.length !== branches.length) throw new Error('merge layer input and parameters should be the same length');

  let _runBranch = (x_branch, layers_branch) => {
    for (let layer of layers_branch) {
      let { layerName, parameters } = layer;
      x_branch = layerFuncs[layerName](arrayType, x_branch, ...parameters);
    }
    return x_branch;
  };

  let y_branches = [];
  for (let i = 0; i < x_branches.length; i++) {
    y_branches.push(_runBranch(x_branches[i], branches[i]));
  }

  let y;
  switch (mode) {
    case 'sum':
      y = ndarray(new arrayType(y_branches[0].size), y_branches[0].shape);
      for (let y_branch of y_branches) {
        ops.addeq(y, y_branch);
      }
      return y;
    case 'ave':
      y = ndarray(new arrayType(y_branches[0].size), y_branches[0].shape);
      for (let y_branch of y_branches) {
        ops.addeq(y, y_branch);
      }
      ops.divseq(y, y_branches.length);
      return y;
    case 'mul':
      y = ndarray(new arrayType(y_branches[0].size), y_branches[0].shape);
      ops.assigns(y, 1.0);
      for (let y_branch of y_branches) {
        ops.muleq(y, y_branch);
      }
      return y;
    case 'concat':
      if (concat_axis < 0) {
        concat_axis += y_branches[0].shape.length;
      }
      if (y_branches[0].shape.length === 1) {
        let newShape = y_branches[0].shape.slice();
        newShape[concat_axis] = 0;
        for (let y_branch of y_branches) {
          newShape[concat_axis] += y_branch.shape[concat_axis];
        }
        let newSize = newShape[0];
        y = ndarray(new arrayType(newSize), newShape);
        y_branches.forEach((y_branch, i) => {
          ops.assign(y.hi((i+1)*y_branch.shape[concat_axis]).lo(i*y_branch.shape[concat_axis]), y_branch);
        });
      } else if (y_branches[0].shape.length === 2) {
        let newShape = y_branches[0].shape.slice();
        newShape[concat_axis] = 0;
        for (let y_branch of y_branches) {
          newShape[concat_axis] += y_branch.shape[concat_axis];
        }
        let newSize = newShape[0] * newShape[1];
        y = ndarray(new arrayType(newSize), newShape);
        y_branches.forEach((y_branch, i) => {
          if (concat_axis === 0) {
            ops.assign(y.hi((i+1)*y_branch.shape[0], y.shape[1]).lo(i*y_branch.shape[0], 0), y_branch);
          } else if (concat_axis === 1) {
            ops.assign(y.hi(y.shape[0], (i+1)*y_branch.shape[1]).lo(0, i*y_branch.shape[1]), y_branch);
          }
        });
      } else if (y_branches[0].shape.length === 3) {
        let newShape = y_branches[0].shape.slice();
        newShape[concat_axis] = 0;
        for (let y_branch of y_branches) {
          newShape[concat_axis] += y_branch.shape[concat_axis];
        }
        let newSize = newShape[0] * newShape[1] * newShape[2];
        y = ndarray(new arrayType(newSize), newShape);
        y_branches.forEach((y_branch, i) => {
          if (concat_axis === 0) {
            ops.assign(y.hi((i+1)*y_branch.shape[0], y.shape[1], y.shape[2]).lo(i*y_branch.shape[0], 0, 0), y_branch);
          } else if (concat_axis === 1) {
            ops.assign(y.hi(y.shape[0], (i+1)*y_branch.shape[1], y.shape[2]).lo(0, i*y_branch.shape[1], 0), y_branch);
          } else if (concat_axis === 2) {
            ops.assign(y.hi(y.shape[0], y.shape[1], (i+1)*y_branch.shape[2]).lo(0, 0, i*y_branch.shape[2]), y_branch);
          }
        });
      } else {
        throw new Error('array shape error in merge layer');
      }
      return y;
    default:
      throw new Error('mode not supported');
  }
}
