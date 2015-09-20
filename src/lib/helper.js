import ndarray from 'ndarray';
import cwise from 'cwise';

let elemAdd = cwise({
  args: ['array', 'array'],
  body: function(a, b) {
    a += b;
  }
});

export { elemAdd };
