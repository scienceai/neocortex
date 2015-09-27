/***************************************************************
Matrix multiplication up to 2048 x 2048 on GPU using WebGL

Matrices are encoded as float32 in textures, then read back out
using readPixels and converted to float32 as ndarray.
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

    this.vertexShaderCode = null;
    this.fragmentShaderCode = null;

  }

  init() {
    if (!this.vertexShaderCode) {
      this.vertexShaderCode = fs.readFileSync('./vertexShader.glsl');
    }
    if (!this.fragmentShaderCode) {
      this.fragmentShaderCode = fs.readFileSync('./fragmentShader.glsl');
    }

    this.canvas = document.getElementById('matmulWebGL');
    if (!this.canvas) {
      let canvas = document.createElement('canvas');
      canvas.id = 'matmulWebGL';
      document.body.appendChild(canvas);
      this.canvas = canvas;
    }
    this.canvas.height = 1;
    this.canvas.width = 1;

    if (!this.GL) {
      let ctxAttrs = { premultipliedAlpha: false, preserveDrawingBuffer: false };
      this.GL = this.canvas.getContext('webgl', ctxAttrs) ||
                this.canvas.getContext('experimental-webgl', ctxAttrs);

      if (typeof this.GL === 'undefined' || this.GL === null) throw new Error('webGL not supported.');

      let floatTextures = this.GL.getExtension('OES_texture_float');
      if (!floatTextures) {
        console.warn('no floating point texture support in WebGL');
      }
    }
    this.GL.viewport(0, 0, 1, 1);

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
        texels[(col * rB + row) * 3 + 1] = this.matB.data[col * rB + row];
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
    let sampler = this.GL.getUniformLocation(this.renderer, 'u_sampler');
		this.GL.uniform1i(sampler, 0);
	}

  _bindUniform() {
    let length = this.GL.getUniformLocation(this.renderer, 'u_length');
    let rowOut = this.GL.getUniformLocation(this.renderer, 'u_rowOuts');
    let colOut = this.GL.getUniformLocation(this.renderer, 'u_colOuts');
    let stepS = this.GL.getUniformLocation(this.renderer, 'u_rowStep');
    let stepT = this.GL.getUniformLocation(this.renderer, 'u_colStep');

    this.GL.uniform1i(length, this.matA.shape[1]);
    this.GL.uniform1f(rowOut, this.matA.shape[0]);
    this.GL.uniform1f(colOut, this.matB.shape[1]);

    // (a x b) x (c x d) -> a x d output, b x c input texture
    this.GL.uniform1f(stepS, 1. / Math.max(this.matA.shape[1], this.matB.shape[1]));
    this.GL.uniform1f(stepT, 1. / Math.max(this.matA.shape[0], this.matB.shape[0]));
  }

  _bindVertices() {
    let a_pos = this.GL.getAttribLocation(this.renderer, 'a_pos');
    let vertexBuffer = this.GL.createBuffer();
    this.GL.bindBuffer(this.GL.ARRAY_BUFFER, vertexBuffer);
    let vertices = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
    this.GL.bufferData(this.GL.ARRAY_BUFFER, new Float32Array(vertices), this.GL.STATIC_DRAW);
    this.GL.vertexAttribPointer(a_pos, 3, this.GL.FLOAT, false, 0, 0);
    this.GL.enableVertexAttribArray(a_pos);

    let a_tex = this.GL.getAttribLocation(this.renderer, 'a_tex');
    let texCoordinates = this.GL.createBuffer();
    this.GL.bindBuffer(this.GL.ARRAY_BUFFER, texCoordinates);
    let textureCoords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    this.GL.bufferData(this.GL.ARRAY_BUFFER, new Float32Array(textureCoords), this.GL.STATIC_DRAW);
    this.GL.vertexAttribPointer(a_tex, 2, this.GL.FLOAT, false, 0, 0);
    this.GL.enableVertexAttribArray(a_tex);

    let indices = this.GL.createBuffer();
    this.GL.bindBuffer(this.GL.ELEMENT_ARRAY_BUFFER, indices);
    let vertexIndices = [0, 1, 2, 0, 2, 3];
    this.GL.bufferData(this.GL.ELEMENT_ARRAY_BUFFER, new Uint16Array(vertexIndices), this.GL.STATIC_DRAW);
  }

  _bindDestTexture() {
    this.destTexture = this.GL.createTexture();
    this.GL.activeTexture(this.GL.TEXTURE1);
    this.GL.bindTexture(this.GL.TEXTURE_2D, this.destTexture);
    this.GL.texImage2D(this.GL.TEXTURE_2D, 0, this.GL.RGBA, this.GL.RGBA, this.GL.UNSIGNED_BYTE, this.canvas);
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

  multiply(matA, matB) {
    this.matA = matA;
    this.matB = matB;
    this.canvas.height = matA.shape[0];
    this.canvas.width = matB.shape[1];
    this.GL.viewport(0, 0, matB.shape[1], matA.shape[0]);

    this._bindTexture();
    this._bindUniform();
    this._bindVertices();
    this._bindDestTexture();
    this._bindFrameBuffer();

    this.GL.drawElements(this.GL.TRIANGLES, 6, this.GL.UNSIGNED_SHORT, 0);

    let buffer = new ArrayBuffer(this.matA.shape[0] * this.matB.shape[1] * 4);
    let product = new Uint8Array(buffer);
    this.GL.readPixels(0, 0, this.matB.shape[1], this.matA.shape[0], this.GL.RGBA, this.GL.UNSIGNED_BYTE, product);

    return ndarray(new Float32Array(buffer), [this.matA.shape[0], this.matB.shape[1]]);
  }

}
