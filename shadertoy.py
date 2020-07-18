# -*- coding: utf-8 -*-
import argparse
import ctypes
import json
import os
import sys

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np


def compile_shader(shader):
	gl.glCompileShader(shader)
	if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
		error = gl.glGetShaderInfoLog(shader).decode()
		raise RuntimeError("Shader not compiled: %s" % error)


def link_shader(program):
	gl.glLinkProgram(program)
	if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
		error = gl.glGetProgramInfoLog(program).decode()
		raise RuntimeError("Shader not linked: %s" % error)


class RenderPass(object):
	def __init__(self, shader, pass_definition):
		self.shader = shader
		self.code = pass_definition['code']
		self.outputs = pass_definition['outputs']
		self.inputs = pass_definition['inputs']
		self.name = pass_definition['name']
		self.type = pass_definition['type']
		if len(self.outputs) != 1:
			raise NotImplementedError("Pass with %d outputs not implemented" % len(outputs))

	@staticmethod
	def create(shader, pass_definition):
		if pass_definition['type'] == 'image':
			return ImageRenderPass(shader, pass_definition)
		else:
			raise NotImplementedError("Shader pass %s not implemented" % pass_definition['type'])

	def render(self):
		raise NotImplementedError()


class ImageRenderPass(RenderPass):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.framebuffer = gl.glGenFramebuffers(1)
		self.image = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.shader.window_size[0], self.shader.window_size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);

	def render(self):
		pass


class Shader(object):
	vertex_shader_code = """attribute vec2 position;
varying vec2 texcoord;
void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
	texcoord = position * vec2(0.5, -0.5) + vec2(0.5);
}"""
	fragment_shader_code = """uniform vec4 color;
void main() { gl_FragColor = color; }"""

	def __init__(self, shader_definition, window_size=None):
		self.vertex_surface = np.array([(-1,+1), (+1,+1), (-1,-1), (+1,-1)], np.float32)
		self.vertex_surface_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_surface.nbytes, self.vertex_surface, gl.GL_DYNAMIC_DRAW)

		self.shader = {
			'program': gl.glCreateProgram(),
			'vertex': gl.glCreateShader(gl.GL_VERTEX_SHADER),
			'fragment': gl.glCreateShader(gl.GL_FRAGMENT_SHADER),
		}
		gl.glShaderSource(self.shader['vertex'], self.vertex_shader_code)
		gl.glShaderSource(self.shader['fragment'], self.fragment_shader_code)
		compile_shader(self.shader['vertex'])
		compile_shader(self.shader['fragment'])
		gl.glAttachShader(self.shader['program'], self.shader['vertex'])
		gl.glAttachShader(self.shader['program'], self.shader['fragment'])
		link_shader(self.shader['program'])
		gl.glDetachShader(self.shader['program'], self.shader['vertex'])
		gl.glDetachShader(self.shader['program'], self.shader['fragment'])

		gl.glUseProgram(self.shader['program'])
		stride = self.vertex_surface.strides[0]
		offset = ctypes.c_void_p(0)
		loc = gl.glGetAttribLocation(self.shader['program'], "position")
		gl.glEnableVertexAttribArray(loc)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)


		loc = gl.glGetUniformLocation(self.shader['program'], "color")
		gl.glUniform4f(loc, 0.0, 0.0, 1.0, 1.0)

		self.window_size = window_size or (512, 512)
		self.render_passes = []
		self.output = None
		for render_pass_definition in shader_definition['renderpass']:
			render_pass = RenderPass.create(self, render_pass_definition)
			if render_pass.type == 'image':
				self.output = render_pass
			self.render_passes.append(render_pass)
		if self.output is None:
			raise RuntimeError("Output is not defined")

		self.frame = 0

	def display(self):
		self.frame += 1
		sys.stdout.write(f'\r{self.frame}')
		sys.stdout.flush()
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		gl.glUseProgram(self.shader['program'])
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
		glut.glutSwapBuffers()

	def reshape(self, width=0, height=0):
		for render_pass in self.render_passes:
			render_pass.render()
		glut.glutReshapeWindow(*self.window_size);
		gl.glViewport(0, 0, *self.window_size)

	def keyboard(self, key, x, y):
		if key == b'\x1b':
			sys.exit()


def main():
	parser = argparse.ArgumentParser(description="Shadertoy renderer")
	parser.add_argument('file', type=argparse.FileType('r'), help="Shader file")
	args = parser.parse_args()

	os.environ['vblank_mode'] = '0'

	glut.glutInit()
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
	glut.glutCreateWindow("Shadertoy")

	shader = Shader(json.load(args.file))
	glut.glutDisplayFunc(shader.display)
	glut.glutReshapeFunc(shader.reshape)
	glut.glutKeyboardFunc(shader.keyboard)
	glut.glutIdleFunc(shader.display)
	shader.reshape()

	glut.glutMainLoop()


if __name__ == "__main__":
	main()
