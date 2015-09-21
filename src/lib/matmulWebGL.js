/***************************************************************
Matrix multiplication up to 2048 x 2048 on GPU using WebGL

Matrices are encoded as float32 in textures, then read back out
using readPixels.
***************************************************************/

import ndarray from 'ndarray';

export default class MatmulWebGL {
  constructor(opts) {
    opts = opts || {};

    this.matGPU = null;

    this.renderBuffer = null;
    this.frameBuffer = null;

    this.vertexShader = '\n\
// works on texels passed to texture shader \n\
#ifdef GL_ES \n\
	precision highp float; \n\
#endif \n\
\n\
attribute vec3 aPos; \n\
attribute vec2 aTex; \n\
varying vec2   vTex; \n\
void main(void) \n\
{ \n\
	gl_Position = vec4(aPos, 1.0); \n\
	vTex = aTex; \n\
}';

    this.fragmentShader = '\n\
// Fragment shader performs calculations based on the passed row and column as texture coordinates. \n\
// Result rendered as 32-bit IEEE754 floating point to the RGBA canvas, readPixel is then used to read out the bytes. \n\
#ifdef GL_ES \n\
	precision highp float; \n\
#endif \n\
\n\
varying vec2	  vTex;         // row, column to calculate \n\
uniform sampler2D usampler;		// left in .r, right in .g \n\
uniform int		  uLength;      // r1xc1.r2xc2 => product has r2 (or c1) terms \n\
uniform float	  uStepS;       // increment across source texture \n\
uniform float	  uStepT;       // increment down source texture \n\
uniform float	  uOutRows;     // size of output in rows \n\
uniform float	  uOutCols;     // size of output in columns \n\
\n\
// sum row r x col c \n\
float sumrowcol(float row, float col) { \n\
	float sum = 0.; \n\
	float ss = 0.;                // column on source texture \n\
	float tt = 0.;                // row on source texture \n\
	float r = row * uStepT;       // moving texture coordinate \n\
	float c = col * uStepS;       // moving texture coordinate \n\
	for (int pos = 0; pos < 2048; ++pos) { \n\
		if (pos >= uLength) break; // stop when we multiple a row by a column \n\
		float m1 = texture2D(usampler, vec2(ss, r)).r; \n\
		float m2 = texture2D(usampler, vec2(c, tt)).g; \n\
		sum += (m1 * m2); \n\
		ss += uStepS; \n\
		tt += uStepT; \n\
	} \n\
	return sum; \n\
} \n\
\n\
void main(void) { \n\
\n\
	// get the implied row and column from .s and .t of passed texel \n\
	float col = floor((vTex.s*uOutRows)); \n\
	float row = floor((vTex.t*uOutCols));    \n\
\n\
	// sum row x col for the passed pixel \n\
	float v = sumrowcol(row,col); \n\
\n\
	// Render to IEEE754 Floating Point \n\
	if (v==0.) { \n\
		gl_FragColor = vec4(0.,0.,0.,0.); \n\
		return; \n\
	} \n\
	float a = abs(v); \n\
	float exp = floor(log2(a)); \n\
	float mantissa = (a * pow(2., 23. - exp)); // fill 24 bits \n\
	float mantissa_0_7 = floor(mantissa / 256. / 256.); \n\
	float mantissa_8_15 = mod(floor(mantissa / 256.),256.); \n\
	float mantissa_16_23 = mod(mantissa,256.); \n\
\n\
	highp float sign = 128. - 128. * (a / v); \n\
	highp float e = (sign + exp + 127.) / 510.;\n\
	highp float m1 = (mantissa_0_7 - (128. * (1. - mod(exp + 127., 2.)))) / 255.; \n\
	highp float m2 = (mantissa_8_15) / 255.; \n\
	highp float m3 = (mantissa_16_23 + .5) / 255.; \n\
	gl_FragColor = vec4(m3, m2, m1, e); \n\
}';

  }

  init(shape) {
		let canvas = document.getElementById('matmulWebGL');
		canvas.height = shape[0];
		canvas.width = shape[1];

    if (!this.matGPU) {
      let ctxAttrs = { premultipliedAlpha: false, preserveDrawingBuffer: false };
      this.matGPU = canvas.getContext('webgl', ctxAttrs) ||
                    canvas.getContext('experimental-webgl', ctxAttrs);

      if (typeof this.matGPU === 'undefined' || this.matGPU === null) throw new Error('webGL not supported.');
    }

		this.matGPU.viewport(0, 0, shape[1], shape[0]);
		return this.matGPU;
	}

}
