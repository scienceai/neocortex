/***************************************************************
Matrix multiplication up to 2048 x 2048 on GPU using WebGL

Matrices are encoded as float32 in textures, then read back out
using readPixels.
***************************************************************/

import ndarray from 'ndarray';

export default class MatmulWebGL {
  constructor(opts) {
    opts = opts || {};

    this.matA = null;
    this.matB = null;

    this.canvas = null;
    this.GL = null;
    this.renderer = null;
    this.texture = null;
    this.destTexture = null;
    this.renderbuffer = null;
    this.framebuffer = null;

    this.vertexShaderCode = '\n\
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

    this.fragmentShaderCode = '\n\
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

  init(matA, matB) {
    this.matA = matA;
    this.matB = matB;

    this.canvas = document.getElementById('matmulWebGL');
    this.canvas.height = matA.shape[0];
    this.canvas.width = matB.shape[1];

    if (!this.GL) {
      let ctxAttrs = { premultipliedAlpha: false, preserveDrawingBuffer: false };
      this.GL = this.canvas.getContext('webgl', ctxAttrs) ||
                this.canvas.getContext('experimental-webgl', ctxAttrs);

      if (typeof this.GL === 'undefined' || this.GL === null) throw new Error('webGL not supported.');
    }

    this.GL.viewport(0, 0, matB.shape[1], matA.shape[0]);

    let vertexShader = this.GL.createShader(this.GL.VERTEX_SHADER);
    this.GL.shaderSource(vertexShader, this.vertexShaderCode);
		this.GL.compileShader(vertexShader);

    let fragmentShader = this.GL.createShader(this.GL.FRAGMENT_SHADER);
    this.GL.shaderSource(fragmentShader, this.fragmentShaderCode);
		this.GL.compileShader(fragmentShader);

		this.renderer = this.GL.createProgram();
		this.GL.attachShader(this.renderer, vertexShader);
		this.GL.attachShader(this.renderer, fragmentShader);
		this.GL.linkProgram(this.renderer);
		this.GL.useProgram(this.renderer);

    this._bindTexture();
    this._bindUniform();
		this._bindVertices();
    this._bindFrameBuffer();

  }

  _bindTexture() {

    let rA = this.matA.shape[0]
      , cA = this.matA.shape[1]
      , rB = this.matB.shape[0]
      , cB = this.matB.shape[1];

    let r = Math.max(rA, rB)
      , c = Math.max(cA, cB);

    let texels = new Float32Array(3 * r * c);

    let count = r * c;
    if (rA === rB && cA === cB) {
      let dest = 0
        , srcA = 0
        , srcB = 0;
      do {
        texels[dest++] = this.matA.data[srcA++];
        texels[dest++] = this.matB.data[srcB++];
        dest++;
      } while(--count);
    } else {
      let row = 0
        , col = 0;
      do {
        texels[(row * cA + col) * 3] = this.matA.data[srcA++];
        texels[(col * rB + row) * 3 + 1] = this.matB[col * rB + row];
        if (col >= cA) {
          col = 0;
          row++;
        }
      } while(--count);
    }

		this.texture = this.GL.createTexture();
    this.GL.activeTexture(this.GL.TEXTURE0);
    this.GL.bindTexture(this.GL.TEXTURE_2D, this.texture);
    this.GL.texImage2D(this.GL.TEXTURE_2D, 0, this.GL.RGB, Math.max(cA, cB), Math.max(rA, rB), 0, this.GL.RGB, this.GL.FLOAT, texels);

    this.GL.texParameteri(this.GL.TEXTURE_2D, this.GL.TEXTURE_WRAP_S, this.GL.CLAMP_TO_EDGE);
    this.GL.texParameteri(this.GL.TEXTURE_2D, this.GL.TEXTURE_WRAP_T, this.GL.CLAMP_TO_EDGE);

    this.GL.texParameteri(this.GL.TEXTURE_2D, this.GL.TEXTURE_MAG_FILTER, this.GL.NEAREST);
    this.GL.texParameteri(this.GL.TEXTURE_2D, this.GL.TEXTURE_MIN_FILTER, this.GL.NEAREST);
		this.GL.uniform1i(this.GL.getUniformLocation(this.renderer,'usampler'), 0);

    this.destTexture = this.GL.createTexture();
		this.GL.activeTexture(this.GL.TEXTURE2);
		this.GL.bindTexture(this.GL.TEXTURE_2D, this.destTexture);
    this.GL.texImage2D(this.GL.TEXTURE_2D, 0, this.GL.RGBA, this.GL.RGBA, this.GL.UNSIGNED_BYTE, this.canvas);
	}

  _bindUniform() {
		let length = this.GL.getUniformLocation(this.renderer, 'uLength');
		let rowOut = this.GL.getUniformLocation(this.renderer, 'uOutRows');
		let colOut = this.GL.getUniformLocation(this.renderer, 'uOutCols');
		let stepS = this.GL.getUniformLocation(this.renderer, 'uStepS');
		let stepT = this.GL.getUniformLocation(this.renderer, 'uStepT');

		this.GL.uniform1i(length, this.matA.shape[1]);

		this.GL.uniform1f(rowOut, this.matA.shape[0]);
		this.GL.uniform1f(colOut, this.matB.shape[1]);
		// 3x10 x 10x2 -> 3x2 output, requires 10x10 input texture
		this.GL.uniform1f(stepS, 1. / Math.max(this.matA.shape[1], this.matB.shape[1]));
		this.GL.uniform1f(stepT, 1. / Math.max(this.matA.shape[0], this.matB.shape[0]));
	}

  _bindVertices() {
		let aPos = this.GL.getAttribLocation(this.renderer, 'aPos');
		let vertexBuffer = this.GL.createBuffer();
		this.GL.bindBuffer(this.GL.ARRAY_BUFFER, vertexBuffer);
		let vertices = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
		this.GL.bufferData(this.GL.ARRAY_BUFFER, new Float32Array(vertices), this.GL.STATIC_DRAW);
		this.GL.vertexAttribPointer(aPos, 3, this.GL.FLOAT, false, 0, 0);
		this.GL.enableVertexAttribArray(aPos);

		let aTex = this.GL.getAttribLocation(this.renderer, 'aTex');
		let texCoordinates = this.GL.createBuffer();
		this.GL.bindBuffer(this.GL.ARRAY_BUFFER, texCoordinates);
		let textureCoords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
		this.GL.bufferData(this.GL.ARRAY_BUFFER, new Float32Array(textureCoords), this.GL.STATIC_DRAW);
		this.GL.vertexAttribPointer(aTex, 2, this.GL.FLOAT, false, 0, 0);
		this.GL.enableVertexAttribArray(aTex);

		let indices = this.GL.createBuffer();
		this.GL.bindBuffer(this.GL.ELEMENT_ARRAY_BUFFER, indices);
		let vertexIndices = [0, 1, 2, 0, 2, 3];
		this.GL.bufferData(this.GL.ELEMENT_ARRAY_BUFFER, new Uint16Array(vertexIndices), this.GL.STATIC_DRAW);
	}

  _bindFrameBuffer() {
		this.renderbuffer = this.renderbuffer || this.GL.createRenderbuffer();
		this.GL.bindRenderbuffer(this.GL.RENDERBUFFER, null);
		this.GL.bindRenderbuffer(this.GL.RENDERBUFFER, this.renderbuffer);
		this.GL.renderbufferStorage(this.GL.RENDERBUFFER, this.GL.DEPTH_COMPONENT16, this.matB.shape[1], this.matA.shape[0]);

		this.framebuffer = this.framebuffer || this.GL.createFramebuffer();
		this.GL.bindFramebuffer(this.GL.FRAMEBUFFER, this.framebuffer);
		this.GL.framebufferTexture2D(this.GL.FRAMEBUFFER, this.GL.COLOR_ATTACHMENT0, this.GL.TEXTURE_2D, this.destTexture, 0);
		this.GL.framebufferRenderbuffer(this.GL.FRAMEBUFFER, this.GL.DEPTH_ATTACHMENT, this.GL.RENDERBUFFER, this.renderbuffer);
	}

  multiply() {
    let buffer = new ArrayBuffer(this.matA.shape[0] * this.matB.shape[1] * 4);
		let product = new Uint8Array(buffer);
		this.GL.readPixels(0, 0, this.matB.shape[1], this.matA.shape[0], this.GL.RGBA, this.GL.UNSIGNED_BYTE, product);

    return new Float32Array(buffer);
  }

}
