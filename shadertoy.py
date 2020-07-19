# -*- coding: utf-8 -*-
import argparse
import ctypes
import json
import os
import sys
import time

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np


FRAGMENT_SHADER_TEMPLATE = """#version 130

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform int       iFrame;                // shader playback frame
uniform float     iChannelTime[4];       // channel playback time (in seconds)
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
uniform vec2      iTileOffset;           // offset of tile
%s

%s

void main()
{
	mainImage(gl_FragColor, gl_FragCoord.xy + iTileOffset);
}

"""

IDENTITY_VERTEX_SHADER = """#version 130

attribute vec2 position;

void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

TEXTURE_VERTEX_SHADER = """#version 130

attribute vec2 position;
varying vec2 texcoord;
void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
	texcoord = position * vec2(0.5, 0.5) + vec2(0.5);
}"""

TEXTURE_FRAGMENT_SHADER = """#version 130

varying vec2 texcoord;
uniform sampler2D texture;
void main()
{
	gl_FragColor = texture2D(texture, texcoord);
}
"""


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


def make_shader(source):
	shader = {'program': gl.glCreateProgram()}
	shader = {
		'program': gl.glCreateProgram(),
	}
	if 'vertex' in source:
		shader['vertex'] = gl.glCreateShader(gl.GL_VERTEX_SHADER)
		gl.glShaderSource(shader['vertex'], source['vertex'])
		compile_shader(shader['vertex'])
		gl.glAttachShader(shader['program'], shader['vertex'])
	if 'fragment' in source:
		shader['fragment'] = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
		gl.glShaderSource(shader['fragment'], source['fragment'])
		compile_shader(shader['fragment'])
		gl.glAttachShader(shader['program'], shader['fragment'])
	link_shader(shader['program'])
	if 'vertex' in source:
		gl.glDetachShader(shader['program'], shader['vertex'])
	if 'fragment' in source:
		gl.glDetachShader(shader['program'], shader['fragment'])
	return shader


class RenderPass(object):
	def __init__(self, renderer, pass_definition):
		self.renderer = renderer
		self.outputs = pass_definition['outputs']
		self.inputs = pass_definition['inputs']
		self.name = pass_definition['name']
		self.type = pass_definition['type']
		if len(self.outputs) != 1:
			raise NotImplementedError("Pass with %d outputs not implemented" % len(outputs))
		self.code = pass_definition['code']
		if self.code.startswith('file://'):
			self.__load_code_from_file(self.code[len('file://'):])
		self.shader = self.make_shader()

	@staticmethod
	def create(renderer, pass_definition):
		if pass_definition['type'] == 'image':
			return ImageRenderPass(renderer, pass_definition)
		else:
			raise NotImplementedError("Shader pass %s not implemented" % pass_definition['type'])

	def make_shader(self):
		shader = make_shader({'vertex': self.get_vertex_shader(), 'fragment': self.get_fragment_shader()})
		gl.glUseProgram(shader['program'])

		stride = self.renderer.vertex_surface.strides[0]
		offset = ctypes.c_void_p(0)
		loc = gl.glGetAttribLocation(shader['program'], "position")
		gl.glEnableVertexAttribArray(loc)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.renderer.vertex_surface_buffer)
		gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

		loc = gl.glGetUniformLocation(shader['program'], "iResolution")
		gl.glUniform3f(loc, float(self.renderer.resolution[0]), float(self.renderer.resolution[1]), float(0))

		return shader

	def render(self):
		raise NotImplementedError()

	def get_vertex_shader(self):
		if hasattr(self, 'vertex_shader_code'):
			return self.vertex_shader_code
		else:
			return IDENTITY_VERTEX_SHADER

	def get_fragment_shader(self):
		# samplerCube
		i_channels = ['sampler2D'] * 4
		sampler_code = ''.join(f'uniform {sampler} iChannel{num};\n' for num, sampler in enumerate(i_channels))
		return FRAGMENT_SHADER_TEMPLATE % (sampler_code, self.code)

	def update_inputs(self, inputs):
		gl.glUseProgram(self.shader['program'])
		for key, val in inputs.items():
			if key == 'time':
				loc = gl.glGetUniformLocation(self.shader['program'], "iTime")
				gl.glUniform1f(loc, val)
			else:
				raise RuntimeError("Unknown input: %s" % key)

	def make_framebuffer(self, multisample=False):
		image = gl.glGenTextures(1)
		framebuffer = gl.glGenFramebuffers(1)
		tile_image = gl.glGenTextures(1)
		tile_framebuffer = gl.glGenFramebuffers(1)

		gl.glBindTexture(gl.GL_TEXTURE_2D, image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.renderer.resolution[0], self.renderer.resolution[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer);
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, image, 0);
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, tile_framebuffer);
		if multisample:
			gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, tile_image);
			gl.glTexImage2DMultisample(gl.GL_TEXTURE_2D_MULTISAMPLE, 4, gl.GL_RGBA8, self.renderer.tile_size[0], self.renderer.tile_size[1], gl.GL_FALSE)
			gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, 0);
			gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D_MULTISAMPLE, tile_image, 0);
		else:
			gl.glBindTexture(gl.GL_TEXTURE_2D, tile_image)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.renderer.tile_size[0], self.renderer.tile_size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);
			gl.glBindTexture(gl.GL_TEXTURE_2D, 0);
			gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tile_image, 0);
		return image, framebuffer, tile_image, tile_framebuffer

	def __load_code_from_file(self, filename):
		shader_dirname = os.path.dirname(self.renderer.shader_filename)
		filename = os.path.abspath(os.path.join(shader_dirname, filename))
		with open(filename, 'r') as fp:
			self.code = fp.read()


class ImageRenderPass(RenderPass):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		framebuffer, image, tile_image, tile_framebuffer = self.make_framebuffer(multisample=self.renderer.multisample)
		self.framebuffer = framebuffer
		self.image = image
		self.tile_image = tile_image
		self.tile_framebuffer = tile_framebuffer

	def render(self):
		gl.glUseProgram(self.shader['program'])

		offsetLoc = gl.glGetUniformLocation(self.shader['program'], "iTileOffset")
		if len(self.renderer.tiles) == 1 and not self.renderer.multisample:
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer);
			gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		else:
			for tile in self.renderer.tiles:
				gl.glUniform2f(offsetLoc, float(tile[0]), float(tile[1]))

				gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.tile_framebuffer);
				gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
				gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);

				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.tile_framebuffer)
				gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.framebuffer)
				gl.glBlitFramebuffer(0, 0, tile[2] - tile[0], tile[3] - tile[1], tile[0], tile[1], tile[2], tile[3], gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
				gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

				gl.glFinish()


class Renderer(object):
	def __init__(self, shader_definition, args):
		self.vertex_surface = np.array([(-1,+1), (+1,+1), (-1,-1), (+1,-1)], np.float32)
		self.vertex_surface_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_surface.nbytes, self.vertex_surface, gl.GL_DYNAMIC_DRAW)

		self.shader = make_shader({'vertex': TEXTURE_VERTEX_SHADER, 'fragment': TEXTURE_FRAGMENT_SHADER})

		gl.glUseProgram(self.shader['program'])
		stride = self.vertex_surface.strides[0]
		offset = ctypes.c_void_p(0)
		loc = gl.glGetAttribLocation(self.shader['program'], "position")
		gl.glEnableVertexAttribArray(loc)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

		self.shader_filename = args.file.name
		self.resolution = args.resolution
		self.tile_size = self.resolution
		if args.tile_size:
			self.tile_size = args.tile_size
		self.tiles = self.__calc_tiles()
		self.multisample = args.multisample
		self.render_passes = []
		self.output = None
		self.start_time = time.monotonic()
		for render_pass_definition in shader_definition['renderpass']:
			render_pass = RenderPass.create(self, render_pass_definition)
			if render_pass.type == 'image':
				self.output = render_pass
			self.render_passes.append(render_pass)
		if self.output is None:
			raise RuntimeError("Output is not defined")

		self.frame = 0

	def display(self):
		self.render_frame()

		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		gl.glUseProgram(self.shader['program'])
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)

		gl.glActiveTexture(gl.GL_TEXTURE0);
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.output.image);
		loc = gl.glGetUniformLocation(self.shader['program'], "texture")
		gl.glUniform1i(loc, 0)

		gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
		gl.glFinish()
		glut.glutSwapBuffers()

	def idle(self):
		glut.glutPostRedisplay()

	def reshape(self, width, height):
		gl.glViewport(0, 0, width, height)

	def keyboard(self, key, x, y):
		if key == b'\x1b':
			sys.exit()

	def render_frame(self):
		self.frame += 1
		sys.stdout.write(f'\r{self.frame}')
		sys.stdout.flush()

		for render_pass in self.render_passes:
			render_pass.update_inputs({
				'time': time.monotonic() - self.start_time,
			})
			render_pass.render()

	def __calc_tiles(self):
		tiled_width = (self.resolution[0] + self.tile_size[0] - 1) // self.tile_size[0]
		tiled_height = (self.resolution[1] + self.tile_size[1] - 1) // self.tile_size[1]
		tiles = []
		for x in range(tiled_width):
			for y in range(tiled_height):
				tile = (
					x * self.tile_size[0],
					y * self.tile_size[1],
					min(x * self.tile_size[0] + self.tile_size[0], self.resolution[0]),
					min(y * self.tile_size[1] + self.tile_size[1], self.resolution[1]),
				)
				tiles.append(tile)
		return tiles


def parse_resolution(val):
	msg = "Resulution in wrong format, expected widthxheight e.g. 1920x1080"
	try:
		width, height = val.split('x')
	except ValueError:
		raise argparse.ArgumentTypeError(msg)
	try:
		width = int(width)
		height = int(height)
	except ValueError:
		raise argparse.ArgumentTypeError(msg)
	return (width, height)


def main():
	parser = argparse.ArgumentParser(description="Shadertoy renderer")
	parser.add_argument('file', type=argparse.FileType('r'), help="Shader file")
	parser.add_argument('--resolution', type=parse_resolution, default=(480, 270), help="Resolution")
	parser.add_argument('--tile-size', type=parse_resolution, help="Tile size")
	parser.add_argument('--multisample', action='store_true', help="Multisampling")
	args = parser.parse_args()

	os.environ['vblank_mode'] = '0'

	glut.glutInit()
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
	glut.glutCreateWindow("Shadertoy")

	renderer = Renderer(json.load(args.file), args)
	glut.glutDisplayFunc(renderer.display)
	glut.glutReshapeFunc(renderer.reshape)
	glut.glutKeyboardFunc(renderer.keyboard)
	glut.glutIdleFunc(renderer.idle)
	glut.glutReshapeWindow(*args.resolution);

	glut.glutMainLoop()


if __name__ == "__main__":
	main()
