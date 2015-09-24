// works on texels passed to texture shader

#ifdef GL_ES
  precision highp float;
#endif

attribute vec3 a_pos;
attribute vec2 a_tex;
varying vec2 v_tex;

void main(void) {
  gl_Position = vec4(a_pos, 1.0);
  v_tex = a_tex;
}
