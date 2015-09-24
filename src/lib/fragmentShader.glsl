// Fragment shader performs calculations based on the passed row and column as texture coordinates.
// Result rendered as 32-bit IEEE754 floating point to the RGBA canvas,
// readPixel is then used to read out the bytes.

#ifdef GL_ES
  precision highp float;
#endif

varying vec2 v_tex;
uniform sampler2D u_sampler;
uniform int u_length;
uniform float u_rowStep;
uniform float u_colStep;
uniform float u_rowOuts;
uniform float u_colOuts;

float sumRowCol(float row, float col) {
  float sum = 0.;
  float sourceTexCol = 0.;
  float sourceTexRow = 0.;
  float r = row * u_colStep;
  float c = col * u_rowStep;
  for (int pos = 0; pos < 2048; ++pos) {
    if (pos >= u_length) break;
    float m1 = texture2D(u_sampler, vec2(sourceTexCol, r)).r;
    float m2 = texture2D(u_sampler, vec2(c, sourceTexRow)).g;
    sum += (m1 * m2);
    sourceTexCol += u_rowStep;
    sourceTexRow += u_colStep;
  }
  return sum;
}

void main(void) {

  float col = floor((v_tex.s * u_rowOuts));
  float row = floor((v_tex.t * u_colOuts));

  float v = sumRowCol(row,col);

  // Render to IEEE754 Floating Point
  if (v==0.) {
    gl_FragColor = vec4(0.,0.,0.,0.);
    return;
  }

  float a = abs(v);
  float exp = floor(log2(a));
  float mantissa = (a * pow(2., 23. - exp)); // fill 24 bits
  float mantissa_0_7 = floor(mantissa / 256. / 256.);
  float mantissa_8_15 = mod(floor(mantissa / 256.),256.);
  float mantissa_16_23 = mod(mantissa,256.);

  highp float sign = 128. - 128. * (a / v);
  highp float e = (sign + exp + 127.) / 510.;
  highp float m1 = (mantissa_0_7 - (128. * (1. - mod(exp + 127., 2.)))) / 255.;
  highp float m2 = (mantissa_8_15) / 255.;
  highp float m3 = (mantissa_16_23 + .5) / 255.;

  gl_FragColor = vec4(m3, m2, m1, e);

}
