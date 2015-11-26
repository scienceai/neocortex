import ndarray from 'ndarray';
import cwiseCompiler from 'cwise-compiler';

let doConvert = cwiseCompiler({'args':['array','scalar','index'],'pre':{'body':'{}','args':[],'thisVars':[],'localVars':[]},'body': {'body': '{\nvar _inline_1_v=_inline_1_arg1_,_inline_1_i\nfor(_inline_1_i=0;_inline_1_i<_inline_1_arg2_.length-1;++_inline_1_i) {\n_inline_1_v=_inline_1_v[_inline_1_arg2_[_inline_1_i]]\n}\n_inline_1_arg0_=_inline_1_v[_inline_1_arg2_[_inline_1_arg2_.length-1]]\n}','args':[{'name':'_inline_1_arg0_','lvalue':true,'rvalue':false,'count':1},{'name':'_inline_1_arg1_','lvalue':false,'rvalue':true,'count':1},{'name':'_inline_1_arg2_','lvalue':false,'rvalue':true,'count':4}],'thisVars':[],'localVars':['_inline_1_i','_inline_1_v']},'post':{'body':'{}','args':[],'thisVars':[],'localVars':[]},'funcName':'convert','blockSize':64});

export default function(arrayType, arr, result) {
  let shape = [], c = arr, sz = 1;
  while (c instanceof Array) {
    shape.push(c.length);
    sz *= c.length;
    c = c[0];
  }
  if(shape.length === 0) {
    return ndarray();
  }
  if(!result) {
    result = ndarray(new arrayType(sz), shape);
  }
  doConvert(result, arr);
  return result;
}
