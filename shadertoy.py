# -*- coding: utf-8 -*-
import argparse
import collections
import contextlib
import ctypes
import enum
import json
import logging
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import unicodedata

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np


logger = logging.getLogger(__name__)

NR_CHANNELS = 4


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
	vec4 color = vec4(0.0,0.0,0.0,1.0);
	mainImage(color, gl_FragCoord.xy + iTileOffset);
	color.w = 1.0;
	gl_FragColor = color;
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

FFPROBE_BINARY = 'ffprobe'
FFMPEG_BINARY = 'ffmpeg'
FFMPEG_CMDLINE = '{ffmpeg} -r {framerate} -f rawvideo -s {resolution} -pix_fmt rgb24 -i {input} -vf vflip -an -y -crf 15 -c:v libx264 -pix_fmt yuv420p -preset slow -loglevel error {output}'
FFPROBE_CMDLINE = '{ffprobe} {input} -print_format json -show_format -show_streams -loglevel error'
FFMPEG_VIDEO_SOURCE = '{ffmpeg} -i {input} {filters} -f rawvideo -pix_fmt rgba -loglevel error {output}'

FRAME_DURATION_AVERAGE = 60


def slugify(name, existing_names=None):
	delimiter = '_'
	name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
	name = re.sub(r'[^a-z0-9]+', '_', name).strip(delimiter)
	if existing_names:
		basename = name
		suffix = 0
		while name in existing_names:
			suffix += 1
			name = basename + '_' + str(suffix)
	return name


def to_local_filename(filename, base=None):
	if os.path.isabs(filename):
		return filename
	if base is None:
		base = __file__
	return os.path.abspath(os.path.join(os.path.dirname(base), filename))


def load_from_file(filename, base=None, binary=True):
	with open(to_local_filename(filename, base), 'r' + ('b' if binary else '')) as fp:
		return fp.read()


def build_shell_command(cmd, replacements):
	replacements = {'{'+key+'}': val for key, val in replacements.items()}
	params = shlex.split(cmd)
	new_params = []
	for param in params:
		replacement = replacements.get(param, param)
		if replacement is None:
			continue
		elif isinstance(replacement, list):
			new_params += replacement
		else:
			new_params.append(replacement)
	return new_params


@contextlib.contextmanager
def open_asset(path, base=None):
	if path.startswith('file://'):
		path = to_local_filename(path[len('file://'):], base)
		fp = open(path, 'rb')
		try:
			yield fp
		finally:
			fp.close()
	else:
		raise NotImplementedError()


class MediaSourceType(enum.Enum):
	image = 1
	audio = 2


class MediaSource(object):
	def __init__(self, fp, media_type=MediaSourceType.image, vflip=False):
		self.__fp = fp
		self.__stream_info = None
		self.__vflip = vflip
		self.media_type = media_type
		replacements = {
			'ffprobe': FFPROBE_BINARY,
			'input': '-',
		}
		self.__fp.seek(0)
		cmd = build_shell_command(FFPROBE_CMDLINE, replacements)
		self.__metadata = json.loads(subprocess.check_output(cmd, input=self.__fp.read()))
		self.__ffmpeg = None
		for stream in self.__metadata['streams']:
			if media_type == MediaSourceType.image and stream['codec_type'] == 'video':
				self.__stream_info = stream
				break
			elif media_type == MediaSourceType.audio and stream['codec_type'] == 'audio':
				self.__stream_info = stream
				break
		self.width = 0
		self.height = 0
		if self.__stream_info is not None:
			if media_type == MediaSourceType.image:
				self.width = self.__stream_info['width']
				self.height = self.__stream_info['height']

	def __ffmpeg_init(self):
		if self.__ffmpeg is None:
			replacements = {
				'ffmpeg': FFMPEG_BINARY,
				'input': '-',
				'output': '-',
				'filters': None,
			}
			if self.__vflip:
				replacements['filters'] = ['-vf', 'vflip']
			cmd = build_shell_command(FFMPEG_VIDEO_SOURCE, replacements)
			self.__fp.seek(0)
			self.__ffmpeg = subprocess.Popen(cmd, stdin=self.__fp, stdout=subprocess.PIPE)

	def get_frame(self):
		if self.media_type != MediaSourceType.image:
			raise RuntimeError("Wrong media type")
		self.__ffmpeg_init()
		data_size = self.width * self.height * 4
		data = self.__ffmpeg.stdout.read(data_size)
		if len(data) == data_size:
			self.__ffmpeg.kill()
			self.__ffmpeg = None
			self.__ffmpeg_init()
			data = self.__ffmpeg.stdout.read(data_size)
		return data


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
	def __init__(self, renderer, input_definition):
		self.renderer = renderer
		self.id = input_definition.get('id')
		self.channel = input_definition.get('channel')
		self.filepath = input_definition.get('filepath')
		self.filter = gl.GL_NEAREST
		if input_definition['sampler']['filter'] == 'mipmap':
			self.filter = gl.GL_LINEAR_MIPMAP_LINEAR
		elif input_definition['sampler']['filter'] == 'linear':
			self.filter = gl.GL_LINEAR
		self.wrap = gl.GL_REPEAT
		if input_definition['sampler']['wrap'] == 'clamp':
			self.wrap = gl.GL_CLAMP_TO_EDGE
		self.vflip = input_definition['sampler']['vflip'] == 'true'
		self.srgb = input_definition['sampler']['srgb'] == 'true'

	@staticmethod
	def create(renderer, input_definition):
		if input_definition['type'] == 'texture':
			return TextureInput(renderer, input_definition)
		elif input_definition['type'] == 'cubemap':
			return CubeMapInput(renderer, input_definition)
		elif input_definition['type'] == 'buffer':
			return BufferInput(renderer, input_definition)
		elif input_definition['type'] == 'keyboard':
			return DummyInput(renderer, input_definition['channel'])
		else:
			raise NotImplementedError("Input type %s not implemented" % input_definition['type'])

	def get_texture(self):
		raise NotImplementedError()

	def get_resolution(self):
		return [0, 0, 0]

	def get_channel_time(self):
		return 0.0

	def load_texture(self):
		pass


class DummyInput(Input):
	def __init__(self, renderer, channel):
		self.renderer = renderer
		self.channel = channel
		self.id = None

	def get_texture(self):
		return None


class TextureInput(Input):
	def __init__(self, renderer, input_definition):
		super().__init__(renderer, input_definition)
		self.texture = None
		self.width = 0
		self.height = 0

	def load_texture(self):
		if self.texture is None:
			self.texture = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
			self._setup_sampler(gl.GL_TEXTURE_2D)

			with open_asset(self.filepath, self.renderer.options.shader_filename) as fp:
				src = MediaSource(fp, vflip=self.vflip)
				frame = src.get_frame()
				self.width = src.width
				self.height = src.height
				if src.width != 0 and src.height != 0:
					gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self._internal_format(), src.width, src.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, frame)
					if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR:
						gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

			gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

	def get_resolution(self):
		return [self.width, self.height, 1]

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_2D, self.texture)

	def _setup_sampler(self, texture_type):
		gl.glTexParameteri(texture_type, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR else self.filter)
		gl.glTexParameteri(texture_type, gl.GL_TEXTURE_MIN_FILTER, self.filter)
		gl.glTexParameteri(texture_type, gl.GL_TEXTURE_WRAP_S, self.wrap)
		gl.glTexParameteri(texture_type, gl.GL_TEXTURE_WRAP_T, self.wrap)
		gl.glTexParameteri(texture_type, gl.GL_TEXTURE_WRAP_R, self.wrap)

	def _internal_format(self):
		return gl.GL_SRGB8_ALPHA8 if self.srgb else gl.GL_RGBA8


class CubeMapInput(TextureInput):
	def __init__(self, renderer, input_definition):
		super().__init__(renderer, input_definition)
		self.texture = None
		self.width = 0
		self.height = 0
		self.depth = 0

	def load_texture(self):
		if self.texture is None:
			self.texture = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texture)
			self._setup_sampler(gl.GL_TEXTURE_CUBE_MAP)
			for i in range(6):
				filepath = (f'_{i}' if i else '').join(os.path.splitext(self.filepath))
				with open_asset(filepath, self.renderer.options.shader_filename) as fp:
					src = MediaSource(fp, vflip=self.vflip)
					frame = src.get_frame()
					if i == 0:
						self.depth = src.width
						self.height = src.height
					elif i == 1:
						self.width = src.width
					if src.width != 0 and src.height != 0:
						gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, self._internal_format(), src.width, src.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, frame)
			if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR:
				gl.glGenerateMipmap(gl.GL_TEXTURE_CUBE_MAP)
			gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

	def get_resolution(self):
		return [self.width, self.height, self.depth]

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_CUBE_MAP, self.texture)


class BufferInput(TextureInput):
	def __init__(self, renderer, input_definition):
		super().__init__(renderer, input_definition)
		self.width = self.renderer.options.w
		self.height = self.renderer.options.h
		self.id = input_definition['id']
		self.input_framebuffer = None
		self.framebuffer = 0
		self.texture = 0

	def load_texture(self):
		if self.texture != 0:
			return
		render_pass = self.renderer.get_render_pass_by_id(self.id)
		self.input_framebuffer = render_pass.get_framebuffer()

		self.framebuffer = gl.glGenFramebuffers(1)
		self.texture = gl.glGenTextures(1)

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer);
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.width, self.height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		self._setup_sampler(gl.GL_TEXTURE_2D)
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture, 0);
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

	def get_resolution(self):
		return [self.width, self.height, 1]

	def get_texture(self):
		gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.input_framebuffer)
		gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.framebuffer)
		gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
		gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
		gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
		if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR:
			gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
		return Texture(gl.GL_TEXTURE_2D, self.texture)


class RenderPass(object):
	def __init__(self, renderer, pass_definition):
		self.renderer = renderer
		self.outputs = pass_definition['outputs']
		self.inputs = []
		self.name = pass_definition['name']
		self.type = pass_definition['type']
		if len(self.outputs) != 1:
			raise NotImplementedError("Pass with %d outputs not implemented" % len(outputs))
		self.output_id = self.outputs[0]['id']
		self.code = pass_definition['code']
		if self.code.startswith('file://'):
			self.code = load_from_file(self.code[len('file://'):], self.renderer.options.shader_filename, binary=False)

		self.inputs = []
		input_definitions = {input_definition['channel']: input_definition for input_definition in pass_definition['inputs']}
		for channel in range(NR_CHANNELS):
			if channel in input_definitions:
				self.inputs.append(Input.create(self.renderer, input_definitions[channel]))
			else:
				self.inputs.append(DummyInput(self.renderer, channel))
		self.shader = self.make_shader()

	def get_texture(self):
		raise NotImplementedError()

	def get_framebuffer(self):
		raise NotImplementedError()

	def render(self):
		raise NotImplementedError()

	@staticmethod
	def create(renderer, pass_definition):
		if pass_definition['type'] == 'image':
			return ImageRenderPass(renderer, pass_definition)
		elif pass_definition['type'] == 'buffer':
			return BufferRenderPass(renderer, pass_definition)
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

		gl.glUniform3f(shader.get_uniform("iResolution"), float(self.renderer.options.w), float(self.renderer.options.h), float(0))

		return shader

	def get_vertex_shader(self):
		if hasattr(self, 'vertex_shader_code'):
			return self.vertex_shader_code
		else:
			return IDENTITY_VERTEX_SHADER

	def get_fragment_shader(self):
		# samplerCube
		i_channels = ['sampler2D'] * NR_CHANNELS
		for i_channel in self.inputs:
			if isinstance(i_channel, CubeMapInput):
				i_channels[i_channel.channel] = 'samplerCube'
		sampler_code = ''.join(f'uniform {sampler} iChannel{num};\n' for num, sampler in enumerate(i_channels))
		return FRAGMENT_SHADER_TEMPLATE % (sampler_code, self.renderer.common + '\n' + self.code)

	def update_uniforms(self, inputs):
		self.shader.use()
		for key, val in inputs.items():
			if key == 'time':
				gl.glUniform1f(self.shader.get_uniform("iTime"), val)
			elif key == 'time_delta':
				gl.glUniform1f(self.shader.get_uniform("iTimeDelta"), val)
			elif key == 'frame':
				gl.glUniform1i(self.shader.get_uniform("iFrame"), val)
			elif key == 'channel_time':
				gl.glUniform1fv(self.shader.get_uniform("iChannelTime"), NR_CHANNELS, np.array(val, np.float32).nbytes)
			elif key == 'channel_resolution':
				gl.glUniform3fv(self.shader.get_uniform("iChannelResolution"), NR_CHANNELS, np.array(val, np.float32).nbytes)
			elif key == 'mouse':
				gl.glUniform4f(self.shader.get_uniform("iMouse"), *val)
			else:
				raise RuntimeError("Unknown input: %s" % key)

	def make_tile_framebuffer(self, internal_format=None):
		if internal_format is None:
			internal_format = self._framebuffer_internal_format
		image = gl.glGenTextures(1)
		framebuffer = gl.glGenFramebuffers(1)
		tile_image = gl.glGenTextures(1)
		tile_framebuffer = gl.glGenFramebuffers(1)

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer);
		gl.glBindTexture(gl.GL_TEXTURE_2D, image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self.renderer.options.w, self.renderer.options.h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, image, 0);
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, tile_framebuffer);
		gl.glBindTexture(gl.GL_TEXTURE_2D, tile_image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self.renderer.options.tile_w, self.renderer.options.tile_h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST);
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST);
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tile_image, 0);
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0);

		return image, framebuffer, tile_image, tile_framebuffer

	def bind_inputs(self):
		for input_channel in self.inputs:
			input_channel.load_texture()
			texture = input_channel.get_texture()
			if texture is not None:
				channel = input_channel.channel
				gl.glActiveTexture(gl.GL_TEXTURE0 + channel + 1);
				gl.glBindTexture(*texture);
				gl.glUniform1i(self.shader.get_uniform(f"iChannel{input_channel.channel}"), channel + 1)
				gl.glActiveTexture(gl.GL_TEXTURE0);
				gl.glBindTexture(texture[0], 0);


class ImageRenderPass(RenderPass):
	_framebuffer_internal_format = gl.GL_RGB

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		framebuffer, image, tile_image, tile_framebuffer = self.make_tile_framebuffer()
		self.framebuffer = framebuffer
		self.image = image
		self.tile_image = tile_image
		self.tile_framebuffer = tile_framebuffer

	def render(self):
		self.shader.use()
		self.bind_inputs()

		tiling = len(self.renderer.tiles) > 1
		for tile in self.renderer.tiles:
			gl.glUniform2f(self.shader.get_uniform("iTileOffset"), float(tile[0]), float(tile[1]))

			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.tile_framebuffer if tiling else self.framebuffer);
			gl.glActiveTexture(gl.GL_TEXTURE0);
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.image);
			gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);

			if tiling:
				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.tile_framebuffer)
				gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.framebuffer)
				gl.glBlitFramebuffer(0, 0, tile[2] - tile[0], tile[3] - tile[1], tile[0], tile[1], tile[2], tile[3], gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
				gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

			if not self.renderer.options.benchmark:
				gl.glFinish()

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_2D, self.image)

	def get_framebuffer(self):
		return self.framebuffer


class BufferRenderPass(ImageRenderPass):
	_framebuffer_internal_format = gl.GL_RGBA32F


class RendererOptions(object):
	def __init__(self, args):
		self.file = args.file
		self.shader_filename = args.file.name
		self.resolution = args.resolution
		self.w, self.h = self.resolution
		self.pixels_count = self.w * self.h
		self.tile_size = self.resolution
		if args.tile_size:
			self.tile_size = args.tile_size
		self.tile_w, self.tile_h = self.tile_size
		self.fps = args.fps
		self.render_video = args.render_video
		self.render_video_fps = args.render_video_fps
		self.benchmark = args.benchmark
		if (args.render_video or self.benchmark) and self.fps is None:
			self.fps = 60
		if args.render_video and self.render_video_fps is None:
			self.render_video_fps = self.fps
		self.quiet = args.quiet


class Renderer(object):
	def __init__(self, shader_definition, options):
		self.options = options
		self.vertex_surface = np.array([(-1,+1), (+1,+1), (-1,-1), (+1,-1)], np.float32)
		self.vertex_surface_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_surface.nbytes, self.vertex_surface, gl.GL_DYNAMIC_DRAW)

		self.shader = Shader(TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER)

		self.shader.use()
		stride = self.vertex_surface.strides[0]
		offset = ctypes.c_void_p(0)
		gl.glBindVertexArray(gl.glGenVertexArrays(1))
		gl.glEnableVertexAttribArray(self.shader.get_attribute("position"))
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glVertexAttribPointer(self.shader.get_attribute("position"), 2, gl.GL_FLOAT, False, stride, offset)

		self.image_pack_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.image_pack_buffer)
		gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self.options.pixels_count * 3, None, gl.GL_STREAM_READ)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
		self.current_image = np.empty((self.options.w, self.options.h, 3), dtype=np.uint8)
		self.cumulative_image = np.zeros((self.options.w, self.options.h, 3), dtype=np.uint32)

		self.video_framerate_controller = None
		if self.options.render_video:
			self.video_framerate_controller = FrameRateController(self.options.fps, self.options.render_video_fps)
		self.tiles = self.__calc_tiles()
		self.render_passes = []
		self.render_passes_by_id = {}
		self.output = None
		self.common = ''
		self.ffmpeg = None
		self.ffmpeg_feed = None
		self.start_time = time.monotonic()
		self.current_time = self.start_time
		self.last_time = self.start_time
		self.frame_durations = []
		self.mouse_state = [0, 0, 0, 0]
		self.mouse_pressed = False

		for render_pass_definition in shader_definition['renderpass']:
			if render_pass_definition['type'] == 'common':
				self.common = render_pass_definition['code']

		for render_pass_definition in shader_definition['renderpass']:
			if render_pass_definition['type'] == 'common':
				continue
			render_pass = RenderPass.create(self, render_pass_definition)
			if render_pass.type == 'image':
				self.output = render_pass
			self.render_passes.append(render_pass)
			self.render_passes_by_id[render_pass.output_id] = render_pass

		if self.output is None:
			raise RuntimeError("Output is not defined")

		self.frame = 0

	def display(self):
		self.render_offscreen()
		self.render_display()

	def render_offscreen(self):
		try:
			gl.glViewport(0, 0, self.options.w, self.options.h)
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

	def mouse(self, button, state, x, y):
		self.mouse_pressed = state == glut.GLUT_DOWN
		coords = self.__transform_mouse_coords(x, y)
		mouse_pressed = 1 if self.mouse_pressed else -1
		self.mouse_state = [coords[0], coords[1], coords[0] * mouse_pressed, coords[1] * mouse_pressed]

	def mouse_motion(self, x, y):
		if not self.mouse_pressed:
			return
		coords = self.__transform_mouse_coords(x, y)
		self.mouse_state[0] = coords[0]
		self.mouse_state[1] = coords[1]

	def render_frame(self):
		uniforms = {}
		if self.options.render_video is not None or self.options.benchmark:
			uniforms['time'] = self.frame / self.options.fps
			uniforms['time_delta'] = 1.0 / self.options.fps
		else:
			uniforms['time'] = self.current_time - self.start_time
			uniforms['time_delta'] = self.current_time - self.last_time
		uniforms['frame'] = self.frame
		uniforms['mouse'] = self.mouse_state

		for render_pass in self.render_passes:
			uniforms['channel_time'] = [channel.get_channel_time() for channel in render_pass.inputs]
			uniforms['channel_resolution'] = [channel.get_resolution() for channel in render_pass.inputs]
			render_pass.update_uniforms(uniforms)

		frame_start = time.monotonic()
		for render_pass in self.render_passes:
			render_pass.render()
		gl.glFinish()
		frame_end = time.monotonic()
		self.frame_durations = self.frame_durations[:FRAME_DURATION_AVERAGE-1] + [frame_end - frame_start]

		fps = 1.0 / statistics.mean(self.frame_durations)
		fps_text = f'{fps:8.8}'
		hour = uniforms['time'] // 3600
		minute = (uniforms['time'] - hour * 3600) // 60
		second = (uniforms['time'] - hour * 3600 - minute * 60)

		if not self.options.quiet:
			sys.stdout.write(f"\x1b[2K\rf: {self.frame:<5} fps: {fps_text:<10} time: {hour}:{minute:02}:{int(second):02}.{int(second*100%100):02}")
			sys.stdout.flush()

		self.frame += 1
		self.last_time = self.current_time
		self.current_time = time.monotonic()

	def write_video(self):
		if not self.options.render_video:
			return
		if self.ffmpeg_feed is None:
			self.__init_video_output()

		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.image_pack_buffer)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.output.framebuffer);
		gl.glReadPixels(0, 0, self.options.w, self.options.h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 0);

		buffer_addr = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
		ctypes.memmove(self.current_image.ctypes.data, buffer_addr, self.options.pixels_count * 3)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0);
		gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

		frame_action = self.video_framerate_controller.on_frame()

		self.cumulative_image += self.current_image
		if frame_action.emit_frames:
			self.current_image = (self.cumulative_image / frame_action.merge_frames).astype(np.uint8)
			for __ in range(frame_action.emit_frames):
				self.ffmpeg_feed.write(self.current_image.tobytes())
			self.cumulative_image = np.zeros((self.options.w, self.options.h, 3), dtype=np.uint32)

	def quit(self):
		if self.ffmpeg is not None:
			self.ffmpeg.stdin.close()
			self.ffmpeg.wait()

	def get_render_pass_by_id(self, output_id):
		return self.render_passes_by_id[output_id]

	def __calc_tiles(self):
		tiled_width = (self.options.w + self.options.tile_w - 1) // self.options.tile_w
		tiled_height = (self.options.h + self.options.tile_h - 1) // self.options.tile_h
		tiles = []
		for x in range(tiled_width):
			for y in range(tiled_height):
				tile = (
					x * self.options.tile_w,
					y * self.options.tile_h,
					min(x * self.options.tile_w + self.options.tile_w, self.options.w),
					min(y * self.options.tile_h + self.options.tile_h, self.options.h),
				)
				tiles.append(tile)
		return tiles

	def __init_video_output(self):
		self.ffmpeg_feed = False
		replacements = {
			'ffmpeg': FFMPEG_BINARY,
			'resolution': f'{self.options.w}x{self.options.h}',
			'input': '-',
			'framerate': str(self.options.render_video_fps),
			'output': self.options.render_video,
		}
		cmd = build_shell_command(FFMPEG_CMDLINE, replacements)
		self.ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
		self.ffmpeg_feed = self.ffmpeg.stdin

	def __transform_mouse_coords(self, x, y):
		window_width = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
		window_height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)
		y = window_height - y
		return (x * self.options.w // window_width, y * self.options.h // window_height)


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


def render(args):
	options = RendererOptions(args)

	os.environ['vblank_mode'] = '0'

	glut.glutInit()
	#glut.glutInitContextVersion(3, 3)
	#glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE)
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
	glut.glutInitWindowSize(*options.resolution);
	glut.glutCreateWindow("Shadertoy")

	renderer = Renderer(json.load(options.file), options)
	glut.glutDisplayFunc(renderer.render_display)
	#glut.glutReshapeFunc(renderer.reshape)
	glut.glutMouseFunc(renderer.mouse)
	glut.glutMotionFunc(renderer.mouse_motion)
	glut.glutKeyboardFunc(renderer.keyboard)
	#glut.glutIdleFunc(renderer.idle)
	glut.glutReshapeWindow(*options.resolution);

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


def extract_sources(args):
	args.file.seek(0)
	shader = json.load(args.file)
	tmp_filename = args.file.name + '.tmp'
	existing_names = set()
	try:
		for renderpass in shader['renderpass']:
			if renderpass['code'].startswith('file://'):
				continue
			basename = slugify(renderpass['name'], existing_names)
			existing_names.add(basename)
			filename = to_local_filename(basename + '.vert', args.file.name)
			with open(filename, 'w') as fp:
				fp.write(renderpass['code'])
			renderpass['code'] = 'file://' + basename + '.vert'
		with open(tmp_filename, 'w') as fp:
			json.dump(shader, fp, indent='\t')
		os.rename(tmp_filename, args.file.name)
	except Exception:
		try:
			os.remove(tmp_filename)
		except FileNotFoundError:
			pass
		raise


def main():
	parser = argparse.ArgumentParser(description="Shadertoy tool")
	subparsers = parser.add_subparsers(help="Command", dest='command')
	subparsers.required = True

	parser_render = subparsers.add_parser('render', help="Render")
	parser_render.add_argument('file', type=argparse.FileType('r'), help="Shader file")
	parser_render.add_argument('--resolution', type=parse_resolution, default=(480, 270), help="Resolution")
	parser_render.add_argument('--tile-size', type=parse_resolution, help="Tile size")
	parser_render.add_argument('--fps', type=int, help="Render frame rate")
	parser_render.add_argument('--render-video', help="Path to rendered video file")
	parser_render.add_argument('--render-video-fps', type=int, help="Video frame rate")
	parser_render.add_argument('--benchmark', action='store_true', help="Benchmark")
	parser_render.add_argument('--quiet', action='store_true', help="Queit")

	parser_extract_sources = subparsers.add_parser('extract_sources', help="Extract shader sources")
	parser_extract_sources.add_argument('file', type=argparse.FileType('r'), help="Shader file")

	args = parser.parse_args()

	globals()[args.command](args)


if __name__ == "__main__":
	main()
