# -*- coding: utf-8 -*-
import argparse
import collections
import ctypes
import json
import logging
import os
import shlex
import subprocess
import sys
import time

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np


logger = logging.getLogger(__name__)


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

FFMPEG_CMDLINE = '{ffmpeg} -r {framerate} -f rawvideo -s {resolution} -pix_fmt rgb24 -i {input} -vf vflip -an -y -crf 15 -c:v libx264 -pix_fmt yuv420p -preset slow {output}'


def to_local_filename(filename, base=None):
	if os.path.isabs(filename):
		return filename
	if base is None:
		base = __file__
	return os.path.abspath(os.path.join(os.path.dirname(base), filename))


def load_from_file(filename, base=None, binary=True):
	with open(to_local_filename(filename, base), 'r' + ('b' if binary else '')) as fp:
		return fp.read()


class Shader(object):
	def __init__(self, vertex, fragment, base=None):
		self.__base = base
		f_prefix = 'file://'
		if vertex.startswith(f_prefix):
			vertex = self.__load_code_from_file(vertex[len(f_prefix):])
		if fragment.startswith(f_prefix):
			fragment = self.__load_code_from_file(fragment[len(f_prefix):])
		self.program = gl.glCreateProgram()
		vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
		fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
		self.__compile_shader(vertex_shader, vertex)
		self.__compile_shader(fragment_shader, fragment)
		self.__link_shader()
		gl.glDetachShader(self.program, vertex_shader)
		gl.glDetachShader(self.program, fragment_shader)

	def use(self):
		gl.glUseProgram(self.program)

	def get_attribute(self, name):
		return gl.glGetAttribLocation(self.program, name)

	def get_uniform(self, name):
		return gl.glGetUniformLocation(self.program, name)

	def __compile_shader(self, shader, code):
		gl.glShaderSource(shader, code)
		gl.glCompileShader(shader)
		if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
			error = gl.glGetShaderInfoLog(shader).decode()
			raise RuntimeError("Shader not compiled: %s" % error)
		gl.glAttachShader(self.program, shader)

	def __link_shader(self):
		gl.glLinkProgram(self.program)
		if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
			error = gl.glGetProgramInfoLog(self.program).decode()
			raise RuntimeError("Shader not linked: %s" % error)


class FrameRateControllerResult(object):
	def __init__(self, emit_frames, merge_frames):
		self.emit_frames = emit_frames
		self.merge_frames = merge_frames

	def __repr__(self):
		return f'emit_frames={self.emit_frames}, merge_frames={self.merge_frames}'


class FrameRateController(object):
	def __init__(self, fps, out_fps):
		self.frame = 0
		self.fps = fps
		self.out_fps = out_fps
		self.last_frame = 0
		self.last_output_frame = 0

	def on_frame(self):
		self.frame += 1
		output_frame = (self.out_fps * self.frame) // self.fps
		if output_frame == self.last_output_frame:
			return FrameRateControllerResult(emit_frames=0, merge_frames=0)
		else:
			emit_frames = output_frame - self.last_output_frame
			merge_frames = self.frame - self.last_frame
			self.last_output_frame = output_frame
			self.last_frame = self.frame
			return FrameRateControllerResult(emit_frames=emit_frames, merge_frames=merge_frames)


Texture = collections.namedtuple('Texture', ['texture_type', 'texture_name'])


class Input(object):
	def __init__(self, renderer):
		self.renderer = renderer

	def get_texture(self):
		raise NotImplementedError()

	def render(self):
		raise NotImplementedError()


class RenderPass(Input):
	def __init__(self, renderer, pass_definition):
		super().__init__(renderer)
		self.outputs = pass_definition['outputs']
		self.inputs = []
		self.name = pass_definition['name']
		self.type = pass_definition['type']
		if len(self.outputs) != 1:
			raise NotImplementedError("Pass with %d outputs not implemented" % len(outputs))
		self.output_id = self.outputs[0]['id']
		self.code = pass_definition['code']
		if self.code.startswith('file://'):
			self.code = load_from_file(self.code[len('file://'):], self.renderer.shader_filename, binary=False)
		self.shader = self.make_shader()

	@staticmethod
	def create(renderer, pass_definition):
		if pass_definition['type'] == 'image':
			return ImageRenderPass(renderer, pass_definition)
		else:
			raise NotImplementedError("Shader pass %s not implemented" % pass_definition['type'])

	def make_shader(self):
		shader = Shader(self.get_vertex_shader(), self.get_fragment_shader())
		shader.use()

		stride = self.renderer.vertex_surface.strides[0]
		offset = ctypes.c_void_p(0)
		gl.glEnableVertexAttribArray(shader.get_attribute("position"))
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.renderer.vertex_surface_buffer)
		gl.glVertexAttribPointer(shader.get_attribute("position"), 2, gl.GL_FLOAT, False, stride, offset)

		gl.glUniform3f(shader.get_uniform("iResolution"), float(self.renderer.resolution[0]), float(self.renderer.resolution[1]), float(0))

		return shader

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
		self.shader.use()
		for key, val in inputs.items():
			if key == 'time':
				gl.glUniform1f(self.shader.get_uniform("iTime"), val)
			else:
				raise RuntimeError("Unknown input: %s" % key)

	def make_tile_framebuffer(self):
		image = gl.glGenTextures(1)
		framebuffer = gl.glGenFramebuffers(1)
		tile_image = gl.glGenTextures(1)
		tile_framebuffer = gl.glGenFramebuffers(1)

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer);
		gl.glBindTexture(gl.GL_TEXTURE_2D, image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.renderer.resolution[0], self.renderer.resolution[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, image, 0);
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, tile_framebuffer);
		gl.glBindTexture(gl.GL_TEXTURE_2D, tile_image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.renderer.tile_size[0], self.renderer.tile_size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tile_image, 0);
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

		return image, framebuffer, tile_image, tile_framebuffer


class ImageRenderPass(RenderPass):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		framebuffer, image, tile_image, tile_framebuffer = self.make_tile_framebuffer()
		self.framebuffer = framebuffer
		self.image = image
		self.tile_image = tile_image
		self.tile_framebuffer = tile_framebuffer

	def render(self):
		self.shader.use()

		tiling = len(self.renderer.tiles) > 1
		for tile in self.renderer.tiles:
			gl.glUniform2f(self.shader.get_uniform("iTileOffset"), float(tile[0]), float(tile[1]))

			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.tile_framebuffer if tiling else self.framebuffer);
			gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);

			if tiling:
				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.tile_framebuffer)
				gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.framebuffer)
				gl.glBlitFramebuffer(0, 0, tile[2] - tile[0], tile[3] - tile[1], tile[0], tile[1], tile[2], tile[3], gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
				gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

			gl.glFinish()

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_2D, self.image)


class Renderer(object):
	def __init__(self, shader_definition, args):
		self.vertex_surface = np.array([(-1,+1), (+1,+1), (-1,-1), (+1,-1)], np.float32)
		self.vertex_surface_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_surface.nbytes, self.vertex_surface, gl.GL_DYNAMIC_DRAW)

		self.shader = Shader(TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER)

		self.shader.use()
		stride = self.vertex_surface.strides[0]
		offset = ctypes.c_void_p(0)
		gl.glEnableVertexAttribArray(self.shader.get_attribute("position"))
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glVertexAttribPointer(self.shader.get_attribute("position"), 2, gl.GL_FLOAT, False, stride, offset)

		self.shader_filename = args.file.name
		self.resolution = args.resolution

		self.image_pack_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.image_pack_buffer)
		gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self.resolution[0] * self.resolution[1] * 3, None, gl.GL_STREAM_READ)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
		self.current_image = np.empty((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)
		self.cumulative_image = np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint32)

		self.tile_size = self.resolution
		if args.tile_size:
			self.tile_size = args.tile_size
		self.fps = args.fps
		self.render_video = args.render_video
		self.render_video_fps = args.render_video_fps
		if args.render_video and self.fps is None:
			self.fps = 60
		if args.render_video and self.render_video_fps is None:
			self.render_video_fps = self.fps
		self.video_framerate_controller = None
		if args.render_video:
			self.video_framerate_controller = FrameRateController(self.fps, self.render_video_fps)
		self.tiles = self.__calc_tiles()
		self.render_passes = []
		self.output = None
		self.ffmpeg = None
		self.ffmpeg_feed = None
		self.start_time = time.monotonic()

		for render_pass_definition in shader_definition['renderpass']:
			render_pass = RenderPass.create(self, render_pass_definition)
			if render_pass.type == 'image':
				self.output = render_pass
			self.render_passes.append(render_pass)

		if self.output is None:
			raise RuntimeError("Output is not defined")

		for render_pass, render_pass_definition in zip(self.render_passes, shader_definition['renderpass']):
			inputs = []

		self.frame = 0

	def display(self):
		self.render_offscreen()
		self.render_display()

	def render_offscreen(self):
		try:
			gl.glViewport(0, 0, self.resolution[0], self.resolution[1])
			self.render_frame()
			self.write_video()
		except Exception:
			logger.exception("Exception in display method")
			sys.exit()

	def render_display(self):
		gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		self.shader.use()
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)

		gl.glActiveTexture(gl.GL_TEXTURE0);
		gl.glBindTexture(*self.output.get_texture());
		gl.glUniform1i(self.shader.get_uniform("texture"), 0)

		gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
		gl.glFinish()
		glut.glutSwapBuffers()

	def idle(self):
		glut.glutPostRedisplay()

	def reshape(self, width, height):
		gl.glViewport(0, 0, width, height)

	def keyboard(self, key, x, y):
		if key == b'\x1b':
			self.quit()
			sys.exit()

	def render_frame(self):
		inputs = {}
		if self.fps is None:
			inputs['time'] = time.monotonic() - self.start_time
		else:
			inputs['time'] = self.frame / self.fps

		for render_pass in self.render_passes:
			render_pass.update_inputs(inputs)
			render_pass.render()

		self.frame += 1
		#sys.stdout.write("\r%d" % self.frame)
		#sys.stdout.flush()

	def write_video(self):
		if not self.render_video:
			return
		if self.ffmpeg_feed is None:
			self.__init_video_output()

		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.image_pack_buffer)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.output.framebuffer);
		gl.glReadPixels(0, 0, self.resolution[0], self.resolution[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 0);

		buffer_addr = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
		ctypes.memmove(self.current_image.ctypes.data, buffer_addr, self.resolution[0] * self.resolution[1] * 3)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

		frame_action = self.video_framerate_controller.on_frame()

		self.cumulative_image += self.current_image
		if frame_action.emit_frames:
			self.current_image = (self.cumulative_image / frame_action.merge_frames).astype(np.uint8)
			for __ in range(frame_action.emit_frames):
				self.ffmpeg_feed.write(self.current_image.tobytes())
			self.cumulative_image = np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint32)

	def quit(self):
		if self.ffmpeg is not None:
			self.ffmpeg.stdin.close()
			self.ffmpeg.wait()

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

	def __init_video_output(self):
		self.ffmpeg_feed = False
		replacements = {
			'ffmpeg': 'ffmpeg',
			'resolution': f'{self.resolution[0]}x{self.resolution[1]}',
			'input': '-',
			'framerate': str(self.render_video_fps),
			'output': self.render_video,
		}
		replacements = {'{'+key+'}': val for key, val in replacements.items()}
		params = shlex.split(FFMPEG_CMDLINE)
		params = [replacements.get(param, param) for param in params]
		self.ffmpeg = subprocess.Popen(params, stdin=subprocess.PIPE)
		self.ffmpeg_feed = self.ffmpeg.stdin


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
	parser.add_argument('--fps', type=int, help="Render frame rate")
	parser.add_argument('--render-video', help="Path to rendered video file")
	parser.add_argument('--render-video-fps', type=int, help="Video frame rate")
	args = parser.parse_args()

	os.environ['vblank_mode'] = '0'

	glut.glutInit()
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
	glut.glutInitWindowSize(*args.resolution);
	glut.glutCreateWindow("Shadertoy")

	renderer = Renderer(json.load(args.file), args)
	glut.glutDisplayFunc(renderer.render_display)
	#glut.glutReshapeFunc(renderer.reshape)
	glut.glutKeyboardFunc(renderer.keyboard)
	#glut.glutIdleFunc(renderer.idle)
	glut.glutReshapeWindow(*args.resolution);

	while True:
		try:
			glut.glutMainLoopEvent()
			renderer.render_offscreen()
			glut.glutMainLoopEvent()
			glut.glutPostRedisplay()
		except KeyboardInterrupt:
			sys.stdout.write("Stopping\n")
			sys.stdout.flush()
			renderer.quit()
			sys.exit()


if __name__ == "__main__":
	main()
