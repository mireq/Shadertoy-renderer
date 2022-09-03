#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import array
import collections
import contextlib
import ctypes
import datetime
import enum
import fractions
import hashlib
import io
import json
import logging
import math
import os
import queue
import re
import shlex
import statistics
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request

if '--no-window' in sys.argv:
	os.environ["PYOPENGL_PLATFORM"] = "egl"
import OpenGL.GL as gl
try:
	import OpenGL.GLUT as glut
except NotImplementedError:
	pass



# Using Dave Hoskins' hash functions (https://www.shadertoy.com/view/4djSRW)


logger = logging.getLogger(__name__)

NR_CHANNELS = 4
AUDIO_FRAME_SIZE = 512
AUDIO_FFT_SIZE = 2048
AUDIO_FFT_SMOOTHING = 0.8
AUDIO_BYTES = 2
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'shadertoy')

COMMON_DEFINITIONS = """
#define HW_PERFORMANCE 1
"""
COMMON_INPUTS = """
#define HW_PERFORMANCE 1

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float   xxiTime;                 // shader playback time (in seconds)
uniform float     iTimeDelta;            // render time (in seconds)
uniform int       iFrame;                // shader playback frame
uniform float     iChannelTime[4];       // channel playback time (in seconds)
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4      iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
uniform vec2      iTileOffset;           // offset of tile
"""
SIMPLE_INPUTS = """
#define iTime xxiTime
"""
ANTIALIAS_INPUTS = """
float iTime = 0.0;
"""


RENDER_MAIN_TEMPLATE = """
	vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
	mainImage(color, gl_FragCoord.xy + iTileOffset);
"""
RENDER_MAIN_ANTIALIAS_TEMPLATE = """
	iTime = xxiTime;
	vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
	vec2 _render_aaOffset;
	vec3 _render_rnd = vec3(gl_FragCoord.xy + iTileOffset, iFrame);
	_render_rnd = fract(_render_rnd * .1031);
	_render_rnd += dot(_render_rnd, _render_rnd.yzx + 33.33);
	float _render_timeOffset = (fract((_render_rnd.x + _render_rnd.y) * _render_rnd.z) / {fps} / ({x_f} * {y_f})) * {shutter_speed};

	const float _aaFractX = 1.0 / {x_f};
	const float _aaFractY = 1.0 / {y_f};
	const float _aaFractX2 = _aaFractX / {y_f};
	const float _aaFractY2 = _aaFractY / {x_f};

	for (int i = 0; i < {x}; ++i) {{
		for (int j = 0; j < {y}; ++j) {{
			_render_aaOffset = vec2(
				float(i)*_aaFractX + float({y} - j - 1) * _aaFractY2,
				float(j)*_aaFractY + float(i) * _aaFractX2
			);
			iTime = xxiTime + ({shutter_speed} / {fps}) * float(i * {y} + j) / float({x} * {y}) + _render_timeOffset;
			vec4 passColor = vec4(0.0, 0.0, 0.0, 1.0);
			mainImage(passColor, gl_FragCoord.xy + iTileOffset + _render_aaOffset);
			passColor.w = 1.0;
			color += passColor;
		}}
	}}

	color = color / ({x_f} * {y_f});
"""
IMAGE_FRAGMENT_SHADER_TEMPLATE = """#version glsl_version
{inputs}
{redefinitions}
{sampler}
{definitions}

void mainImage(out vec4 c, in vec2 f);

{common}
{code}

out vec4 outColor;

void main(void)
{{
	{render_main}
	color.w = 1.0;
	outColor = color;
}}
"""
BUFFER_FRAGMENT_SHADER_TEMPLATE = """#version glsl_version
{inputs}
{redefinitions}
{sampler}
{definitions}

void mainImage(out vec4 c, in vec2 f);

{common}
{code}

out vec4 outColor;

void main(void)
{{
	{render_main}
	outColor = color;
}}
"""
SOUND_FRAGMENT_SHADER_TEMPLATE = """#version glsl_version
{inputs}

{redefinitions}
{sampler}
{definitions}

vec2 mainSound(in int samp, float time);

{common}
{code}

out vec4 outColor;

void main(void)
{{
	float t = iTimeOffset + ((gl_FragCoord.x-0.5) + (gl_FragCoord.y-0.5)*512.0)/iSampleRate;
	int s = iSampleOffset + int(gl_FragCoord.y-0.2)*512 + int(gl_FragCoord.x-0.2);
	vec2 y = mainSound(s, t);
	vec2 v = floor((0.5+0.5*y)*65536.0);
	vec2 vl = mod(v,256.0)/255.0;
	vec2 vh = floor(v/256.0)/255.0;
	outColor = vec4(vl.x, vh.x, vl.y, vh.y);
}}
"""

IDENTITY_VERTEX_SHADER = """#version glsl_version

in vec2 position;

void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
}
"""

TEXTURE_VERTEX_SHADER = """#version glsl_version

in vec2 position;
out vec2 texcoord;
void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
	texcoord = position * vec2(0.5, 0.5) + vec2(0.5);
}"""

TEXTURE_FRAGMENT_SHADER = """#version glsl_version

in vec2 texcoord;
uniform sampler2D texture;
out vec4 outColor;
void main()
{
	outColor = texture2D(texture, texcoord);
}
"""
VIDEO_FRAGMENT_SHADER = """#version glsl_version

in vec2 texcoord;
uniform int average_frames;
uniform int current_frame;
uniform int output_frame_number;
uniform int dithering;
uniform sampler2D input_buffer;
uniform sampler2D output_buffer;
out vec4 outColor;

void main()
{
	vec4 color;
	if (current_frame == 0) {
		color = texture2D(input_buffer, texcoord);
	}
	else {
		color = texture2D(input_buffer, texcoord) + texture2D(output_buffer, texcoord);
	}
	if (average_frames > 1) {
		color = color / float(average_frames);
	}

	if (dithering > 0) {
		vec3 rnd = fract(vec3(gl_FragCoord.xy, output_frame_number) * vec3(.1031, .1030, .0973));
		rnd += dot(rnd, rnd.yxz+33.33);
		rnd = fract((rnd.xxy + rnd.yxx)*rnd.zyx) - 0.5;
		float rnd_val = rnd.x + rnd.y + rnd.z;
		rnd = vec3(rnd.x + rnd_val, rnd.y + rnd_val, rnd.z + rnd_val) / (256.0 / float(dithering));
		color = vec4(color.rgb + rnd, color.a);
	}

	outColor = color;
}
"""

YT_DL_BINARY = 'yt-dlp'
FFPROBE_BINARY = 'ffprobe'
FFMPEG_BINARY = 'ffmpeg'
FFMPEG_CMDLINE = '{ffmpeg} -r {framerate} -f rawvideo -s {resolution} -pix_fmt gbrpf32le -color_range pc -color_trc linear -color_primaries bt709 -colorspace bt709 -i {input} {more_inputs} -y -crf {crf} -c:v {codec} -c:a flac -pix_fmt {pix_fmt} -preset {preset} {hdr_args} {extra_args} -loglevel error {output}'
FFPROBE_CMDLINE = '{ffprobe} {input} -print_format json -show_format -show_streams -loglevel error'
FFMPEG_VIDEO_SOURCE = '{ffmpeg} -i {input} {filters} -f rawvideo -pix_fmt rgba -loglevel error {output}'
FFMPEG_AUDIO_SOURCE = '{ffmpeg} -i {input} -ac 1 -f u16le -vn -loglevel error {output}'

FRAME_DURATION_AVERAGE = 60


class Blitter(object):
	def __init__(self, src_framebuffer=None, target_framebuffer=None):
		self.src_framebuffer = src_framebuffer
		self.target_framebuffer = target_framebuffer
		if self.src_framebuffer is None:
			self.src_framebuffer = gl.glGenFramebuffers(1)
		if self.target_framebuffer is None:
			self.target_framebuffer = gl.glGenFramebuffers(1)

	def set_src_texture(self, texture):
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.src_framebuffer)
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

	def set_target_texture(self, texture):
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.target_framebuffer)
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

	def blit(self, src_rect, target_rect):
		gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.src_framebuffer)
		gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.target_framebuffer)
		args = list(src_rect) + list(target_rect) + [gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST]
		gl.glBlitFramebuffer(*args)
		gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
		gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)


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


def get_asset_filename(path, base=None):
	cache_dir = os.path.join(CACHE_DIR, 'assets')
	try:
		os.makedirs(cache_dir)
	except FileExistsError:
		pass
	if path.startswith('file://'):
		return to_local_filename(path[len('file://'):], base)
	elif path.startswith('/'):
		basename = os.path.basename(path)
		prefix = hashlib.sha1(path[:-len(basename)].encode('utf-8')).hexdigest()
		basename = prefix + '_' + basename
		cached_name = os.path.join(cache_dir, basename)
		if os.path.exists(cached_name):
			return cached_name
		urllib.request.urlretrieve('https://www.shadertoy.com' + path, cached_name)
		return cached_name
	elif path.startswith('https://soundcloud.com'):
		cached_name = os.path.abspath(os.path.join(cache_dir, hashlib.sha1(path.encode('utf-8')).hexdigest() + '.audio'))
		if os.path.exists(cached_name):
			return cached_name
		subprocess.call([YT_DL_BINARY, path, '-o', cached_name])
		return cached_name
	else:
		print(path)
		raise NotImplementedError()


@contextlib.contextmanager
def open_asset(path, base=None, leave_open=False):
	cached_name = get_asset_filename(path, base)
	try:
		fp = open(cached_name, 'rb')
		yield fp
	finally:
		if not leave_open:
			fp.close()


class PipeIO(object):
	def __init__(self, r, w, queue_size=1):
		self.r = r
		self.w = w
		self.w_queue = queue.Queue(maxsize=queue_size)
		threading.Thread(target=self.write_worker, daemon=True).start()

	def write_worker(self):
		while True:
			packet = self.w_queue.get()
			if self.w is None:
				continue
			while packet:
				try:
					count = os.write(self.w, packet)
					if count < len(packet):
						packet = packet[count:]
					else:
						break
				except (KeyboardInterrupt, SystemExit):
					if self.w is not None:
						os.close(self.w)
						self.w = None
					raise

	def read(self, size):
		return os.read(self.r, size)

	def write(self, data):
		self.w_queue.put(data)

	def close(self):
		if self.r is not None:
			os.close(self.r)
			self.r = None
		if self.w is not None:
			os.close(self.w)
			self.w = None

	def w_full(self):
		return self.w_queue.full()


class MediaSourceType(enum.Enum):
	image = 1
	audio = 2


class MediaSource(object):
	def __init__(self, fp, renderer, media_type=MediaSourceType.image, vflip=False):
		self.__fp = fp
		self.__renderer = renderer
		self.__stream_info = None
		self.__vflip = vflip
		self.media_type = media_type
		replacements = {
			'ffprobe': FFPROBE_BINARY,
			'input': 'pipe:0',
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
		self.sample_rate = 44100

		self.current_audio_sample = 0
		self.current_audio_buffer = None
		self.current_fft_buffer = b''
		self.current_fft = None
		self.__blackman = None

		if self.__stream_info is not None:
			if media_type == MediaSourceType.image:
				self.width = self.__stream_info['width']
				self.height = self.__stream_info['height']
			elif media_type == MediaSourceType.audio:
				self.sample_rate = int(self.__stream_info['sample_rate'])

	def __ffmpeg_init(self):
		if self.__ffmpeg is None:
			replacements = {
				'ffmpeg': FFMPEG_BINARY,
				'input': 'pipe:0',
				'output': 'pipe:1',
				'filters': None,
			}
			if self.media_type == MediaSourceType.image:
				if self.__vflip:
					replacements['filters'] = ['-vf', 'vflip']
				cmd = build_shell_command(FFMPEG_VIDEO_SOURCE, replacements)
			else:
				replacements.pop('filters')
				cmd = build_shell_command(FFMPEG_AUDIO_SOURCE, replacements)
			self.__fp.seek(0)
			self.__ffmpeg = subprocess.Popen(cmd, stdin=self.__fp, stdout=subprocess.PIPE)

	def get_frame(self):
		self.__ffmpeg_init()
		if self.media_type == MediaSourceType.image:
			data_size = self.width * self.height * 4
			data = self.__ffmpeg.stdout.read(data_size)
			if len(data) == data_size:
				self.__ffmpeg.kill()
				self.__ffmpeg = None
				self.__ffmpeg_init()
				data = self.__ffmpeg.stdout.read(data_size)
		else:
			import numpy as np

			audio_buffer_bytes = AUDIO_FRAME_SIZE * AUDIO_BYTES
			fft_buffer_bytes = AUDIO_FFT_SIZE * AUDIO_BYTES

			requested_sample = math.floor(self.sample_rate * self.__renderer.exact_time)
			additional_size = (requested_sample - self.current_audio_sample) * AUDIO_BYTES
			self.current_audio_sample = requested_sample

			if self.current_audio_buffer is None:
				self.current_audio_buffer = b'\0' * audio_buffer_bytes
			if self.current_fft is None:
				self.current_fft = np.array([-100.0] * AUDIO_FRAME_SIZE)

			if additional_size:
				additional_data = self.__ffmpeg.stdout.read(additional_size)
				if len(additional_data) != additional_size:
					additional_data = additional_data.ljust(additional_size, b'\0')
				self.current_fft_buffer += additional_data
				self.current_audio_buffer = (self.current_audio_buffer + additional_data)[-audio_buffer_bytes:]

			while len(self.current_fft_buffer) >= fft_buffer_bytes:
				fft = self.current_fft_buffer[:fft_buffer_bytes]
				self.current_fft_buffer = self.current_fft_buffer[audio_buffer_bytes:]
				if self.__blackman is None:
					self.__blackman = np.blackman(AUDIO_FFT_SIZE)
				fft = np.frombuffer(fft, dtype=np.uint16).astype(float)
				fft = fft * self.__blackman
				fft = np.fft.rfft(fft)[:AUDIO_FRAME_SIZE] / (AUDIO_FFT_SIZE * 65535 / 2)
				fft = np.abs(fft)
				fft[fft == 0] = 0.0000000001
				fft = 20.*np.log10(np.abs(fft)) # decibels

				self.current_fft = (self.current_fft * AUDIO_FFT_SMOOTHING) + (fft * (1 - AUDIO_FFT_SMOOTHING))

			fft = self.current_fft.copy()
			db_min = -100
			db_max = -30
			fft = (fft - db_min) * (65535.0 / (db_max - db_min))
			data = np.clip(fft, 0, 65535).astype(np.uint16).tobytes() + self.current_audio_buffer
		return data


class Shader(object):
	def __init__(self, vertex, fragment, base=None, name=None):
		self.__base = base
		self.__name = name or '<unnamed>'
		f_prefix = 'file://'
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
			if isinstance(error, bytes):
				error = error.decode()
			line_number = re.match(r'\d+:(\d+)', error)
			if line_number:
				line_number = int(line_number.group(1))
				code = code.splitlines()
				sys.stdout.write('> %s\n' % code[line_number - 1])
				sys.stdout.flush()
			raise RuntimeError("Shader %s not compiled:\n%s" % (self.__name, error))
		gl.glAttachShader(self.program, shader)

	def __link_shader(self):
		gl.glLinkProgram(self.program)
		if not gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
			error = gl.glGetProgramInfoLog(self.program)
			if isinstance(error, bytes):
				error = error.decode()
			raise RuntimeError("Shader %s not linked: %s" % (self.__name, error))


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
		elif input_definition['type'] == 'music':
			return MusicInput(renderer, input_definition)
		elif input_definition['type'] == 'musicstream':
			return MusicStreamInput(renderer, input_definition)
		elif input_definition['type'] == 'cubemap':
			return CubeMapInput(renderer, input_definition)
		elif input_definition['type'] == 'buffer':
			return BufferInput(renderer, input_definition)
		elif input_definition['type'] == 'volume':
			return VolumeInput(renderer, input_definition)
		elif input_definition['type'] == 'keyboard':
			return KeyboardInput(renderer, input_definition)
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
				src = MediaSource(fp, self.renderer, vflip=self.vflip)
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


class MusicInput(TextureInput):
	def __init__(self, renderer, input_definition):
		super().__init__(renderer, input_definition)
		self.texture = None
		self.media_src = None

	def load_texture(self):
		if self.texture is None:
			self.texture = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
			self._setup_sampler(gl.GL_TEXTURE_2D)

			with open_asset(self.filepath, self.renderer.options.shader_filename, leave_open=True) as fp:
				self.media_src = MediaSource(fp, self.renderer, media_type=MediaSourceType.audio)
		frame = self.media_src.get_frame()
		if frame:
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self._internal_format(), AUDIO_FRAME_SIZE, 2, 0, gl.GL_RED, gl.GL_UNSIGNED_SHORT, frame)
			if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR:
				gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

	def get_resolution(self):
		return [AUDIO_FRAME_SIZE, 2, 1]

	def _internal_format(self):
		return gl.GL_R16


class MusicStreamInput(MusicInput):
	pass


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
					src = MediaSource(fp, self.renderer, vflip=self.vflip)
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
		self.texture = 0
		self.blitter = None

	def load_texture(self):
		if self.texture != 0:
			return
		render_pass = self.renderer.get_render_pass_by_id(self.id)

		self.texture = gl.glGenTextures(1)

		self.blitter = Blitter(render_pass.get_framebuffer())

		gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.width, self.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
		self._setup_sampler(gl.GL_TEXTURE_2D)
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

		self.blitter.set_target_texture(self.texture)

	def get_resolution(self):
		return [self.width, self.height, 1]

	def get_texture(self):
		rect = [0, 0, self.width, self.height]
		self.blitter.blit(rect, rect)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
		if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR:
			gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
		return Texture(gl.GL_TEXTURE_2D, self.texture)


class VolumeInput(TextureInput):
	def __init__(self, renderer, input_definition):
		super().__init__(renderer, input_definition)
		self.depth = 0

	def load_texture(self):
		if self.texture is None:
			self.texture = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_3D, self.texture)
			self._setup_sampler(gl.GL_TEXTURE_3D)

			with open_asset(self.filepath, self.renderer.options.shader_filename) as fp:
				signature = fp.read(4)
				self.width, self.height, self.depth = struct.unpack('III', fp.read(12))
				num_channels, __, bin_fmt = struct.unpack('BBH', fp.read(4))
				data = fp.read()
				opengl_type = None
				OPENGL_FORMATS = {
					(0, 1): gl.GL_R8,
					(0, 2): gl.GL_RG8,
					(0, 3): gl.GL_RGB8,
					(0, 4): gl.GL_RGBA8,
					(10, 1): gl.GL_R32F,
					(10, 2): gl.GL_RG32F,
					(10, 3): gl.GL_RGB32F,
					(10, 4): gl.GL_RGBA32F,
				}
				PIXEL_FORMATS = {
					1: gl.GL_RED,
					2: gl.GL_RG,
					3: gl.GL_RGB,
					4: gl.GL_RGBA,
				}
				opengl_format = OPENGL_FORMATS[(bin_fmt, num_channels)]
				pixel_format = PIXEL_FORMATS[num_channels]
				if bin_fmt == 0: # uint
					opengl_type = gl.GL_UNSIGNED_BYTE
				elif bin_fmt == 10: # float
					opengl_type = gl.GL_FLOAT

				gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, opengl_format, self.width, self.height, self.depth, 0, pixel_format, opengl_type, data)
				if self.filter == gl.GL_LINEAR_MIPMAP_LINEAR:
					gl.glGenerateMipmap(gl.GL_TEXTURE_3D)

			gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

	def get_resolution(self):
		return [self.width, self.height, self.depth]

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_3D, self.texture)


class KeyboardInput(Input):
	def __init__(self, renderer, input_definition):
		super().__init__(renderer, input_definition)
		self.texture = None
		self.width = 256
		self.height = 3
		self.keyboard_status = array.array('B', [0] * self.width * self.height)
		self.changed_keyboard = False

	def load_texture(self):
		if self.texture is None:
			self.texture = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
			gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, self.width, self.height, 0, gl.GL_RED, gl.GL_UNSIGNED_BYTE, None)
		if self.changed_keyboard:
			self.changed_keyboard = False
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, self.width, self.height, 0, gl.GL_RED, gl.GL_UNSIGNED_BYTE, self.keyboard_status.tobytes())
			gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

	def get_resolution(self):
		return [self.width, self.height, 1]

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_2D, self.texture)

	def keyboard(self, key, pressed): # pylint: disable=unused-argument
		if not pressed:
			key = ord(key)
			if key <= 255:
				self.keyboard_status[key + 0*256] = 0
				self.keyboard_status[key + 1*256] = 0

		for key in self.renderer.pressed_keys:
			key = ord(key)
			if key > 255:
				continue
			self.keyboard_status[key + 0*256] = 255
			self.keyboard_status[key + 1*256] = 255
			self.keyboard_status[key + 2*256] = 255
		self.changed_keyboard = True


class BaseRenderPass(object):
	def __init__(self, renderer):
		self.inputs = []
		self.renderer = renderer

	def update_uniforms(self, inputs):
		pass

	def destroy(self):
		pass


class RenderPass(BaseRenderPass):
	_fragment_shader_template = None

	def __init__(self, renderer, pass_definition):
		super().__init__(renderer)
		self.outputs = pass_definition['outputs']
		self.name = pass_definition['name']
		self.type = pass_definition['type']
		if len(self.outputs) > 1:
			raise NotImplementedError("Pass with %d outputs not implemented" % len(self.outputs))
		if len(self.outputs) == 1:
			self.output_id = self.outputs[0]['id']
		else:
			self.output_id = None
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
		elif pass_definition['type'] == 'sound':
			return SoundRenderPass(renderer, pass_definition)
		elif pass_definition['type'] == 'video':
			return VideoRenderPass(renderer)
		else:
			raise NotImplementedError("Shader pass %s not implemented" % pass_definition['type'])

	def make_shader(self):
		shader = Shader(
			self.renderer.process_shader(self.get_vertex_shader()),
			self.renderer.process_shader(self.get_fragment_shader()),
			name=self.name
		)
		shader.use()

		#gl.glEnableVertexAttribArray(shader.get_attribute("position"))
		#gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.renderer.vertex_surface_buffer)
		#gl.glVertexAttribPointer(shader.get_attribute("position"), 2, gl.GL_FLOAT, False, self.renderer.vertex_surface_data.itemsize * 2, ctypes.c_void_p(0))

		gl.glUniform3f(shader.get_uniform("iResolution"), float(self.renderer.options.w), float(self.renderer.options.h), float(1))

		return shader

	def get_vertex_shader(self):
		if hasattr(self, 'vertex_shader_code'):
			return self.vertex_shader_code
		else:
			return IDENTITY_VERTEX_SHADER

	def get_fragment_shader_replacements(self):
		# samplerCube
		i_channels = ['sampler2D'] * NR_CHANNELS
		for i_channel in self.inputs:
			if isinstance(i_channel, CubeMapInput):
				i_channels[i_channel.channel] = 'samplerCube'
			elif isinstance(i_channel, VolumeInput):
				i_channels[i_channel.channel] = 'sampler3D'
		sampler_code = ''.join(f'uniform {sampler} iChannel{num};\n' for num, sampler in enumerate(i_channels))
		has_antialias = self.renderer.options.antialias != (1, 1)
		main = RENDER_MAIN_TEMPLATE
		def force_float(val):
			val = str(val)
			if not '.' in val:
				val = val + '.'
			return val
		if has_antialias:
			main = RENDER_MAIN_ANTIALIAS_TEMPLATE.format(
				x=self.renderer.options.antialias[0],
				y=self.renderer.options.antialias[1],
				x_f=force_float(self.renderer.options.antialias[0]),
				y_f=force_float(self.renderer.options.antialias[1]),
				fps=force_float(self.renderer.options.fps),
				shutter_speed=force_float(self.renderer.options.shutter_speed),
			)
		return dict(
			inputs=COMMON_INPUTS,
			definitions=COMMON_DEFINITIONS,
			render_main=main,
			redefinitions=ANTIALIAS_INPUTS if has_antialias else SIMPLE_INPUTS,
			sampler=sampler_code,
			common=self.renderer.common,
			code=self.code
		)

	def get_fragment_shader(self):
		return self._fragment_shader_template.format(**self.get_fragment_shader_replacements())

	def update_uniforms(self, inputs):
		self.shader.use()
		for key, val in inputs.items():
			if key == 'time':
				gl.glUniform1f(self.shader.get_uniform("xxiTime"), val)
			elif key == 'time_delta':
				gl.glUniform1f(self.shader.get_uniform("iTimeDelta"), val)
			elif key == 'frame':
				gl.glUniform1i(self.shader.get_uniform("iFrame"), val)
			elif key == 'channel_time':
				gl.glUniform1fv(self.shader.get_uniform("iChannelTime"), NR_CHANNELS, val)
			elif key == 'channel_resolution':
				gl.glUniform3fv(self.shader.get_uniform("iChannelResolution"), NR_CHANNELS, val)
			elif key == 'mouse':
				gl.glUniform4f(self.shader.get_uniform("iMouse"), *val)
			elif key == 'date':
				gl.glUniform4f(self.shader.get_uniform("iDate"), *val)
			else:
				raise RuntimeError("Unknown input: %s" % key)

	def make_tile_framebuffer(self, internal_format=None):
		if internal_format is None:
			internal_format = self._framebuffer_internal_format
		image, tile_image = gl.glGenTextures(2)
		framebuffer, tile_framebuffer = gl.glGenFramebuffers(2)

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
		gl.glBindTexture(gl.GL_TEXTURE_2D, image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self.renderer.options.w, self.renderer.options.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, image, 0)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, tile_framebuffer)
		gl.glBindTexture(gl.GL_TEXTURE_2D, tile_image)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, self.renderer.options.tile_w, self.renderer.options.tile_h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tile_image, 0)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
		gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

		return image, framebuffer, tile_image, tile_framebuffer

	def bind_inputs(self):
		for input_channel in self.inputs:
			input_channel.load_texture()
			texture = input_channel.get_texture()
			if texture is not None:
				channel = input_channel.channel
				gl.glActiveTexture(gl.GL_TEXTURE0 + channel + 1)
				gl.glBindTexture(*texture)
				gl.glUniform1i(self.shader.get_uniform(f"iChannel{input_channel.channel}"), channel + 1)
				gl.glActiveTexture(gl.GL_TEXTURE0)
				gl.glBindTexture(texture[0], 0)


class ImageRenderPass(RenderPass):
	_framebuffer_internal_format = gl.GL_RGBA32F
	_fragment_shader_template = IMAGE_FRAGMENT_SHADER_TEMPLATE

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		image, framebuffer, tile_image, tile_framebuffer = self.make_tile_framebuffer()
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

			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.tile_framebuffer if tiling else self.framebuffer)
			gl.glActiveTexture(gl.GL_TEXTURE0)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.image)
			self.renderer.enable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
			gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
			self.renderer.disable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

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


class VideoRenderPass(BaseRenderPass):
	_framebuffer_internal_format = gl.GL_RGBA32F

	def __init__(self, renderer):
		super().__init__(renderer)
		self.ffmpeg_feed = False
		more_inputs = []
		audio_inputs = []
		if renderer.sound is not None:
			r, w = os.pipe()
			os.set_inheritable(r, True)
			os.set_inheritable(w, True)
			audio_inputs.append(PipeIO(r, w, 2))
			more_inputs += ['-f', 'u16le', '-sample_rate', str(renderer.sound.sample_rate), '-channels', '2', '-codec:a', 'pcm_u16le', '-channel_layout', 'stereo', '-i', 'pipe:' + str(r)]

		channels_visited = set()
		for render_pass in self.renderer.render_passes:
			for input_channel in render_pass.inputs:
				if input_channel.id in channels_visited:
					continue
				channels_visited.add(input_channel.id)
				if isinstance(input_channel, MusicInput):
					filename = get_asset_filename(input_channel.filepath, self.renderer.options.shader_filename)
					more_inputs += ['-i', filename]

		CODECS = {
			'h264': 'libx264',
			'h265': 'libx265',
		}
		replacements = {
			'ffmpeg': FFMPEG_BINARY,
			'resolution': f'{self.renderer.options.w}x{self.renderer.options.h}',
			'input': 'pipe:0',
			'framerate': str(self.renderer.options.render_video_fps),
			'output': self.renderer.options.render_video,
			'more_inputs': more_inputs,
			'codec': CODECS[self.renderer.options.render_video_codec],
			'preset': self.renderer.options.render_video_preset,
			'crf': str(self.renderer.options.render_video_crf),
			'pix_fmt': str(self.renderer.options.render_video_pix_fmt),
			'hdr_args': ''
		}
		if self.renderer.options.render_video_hdr:
			#replacements['hdr_args']=['-vf', 'colorspace=bt2020:ispace=bt709:itrc=linear:iprimaries=bt709:trc=bt2020-12:range=pc:format=yuv444p12', '-color_range', 'pc', '-color_trc', 'arib-std-b67', '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc', '-x265-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400', '-x265-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400']
			#replacements['hdr_args']=['-vf', 'zscale=rin=full:pin=709:tin=linear:min=709:r=full:p=2020:t=linear,format=yuv444p12le', '-color_range', 'pc', '-color_trc', 'arib-std-b67', '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc', '-x265-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full:max-cll=203,203']
			#replacements['hdr_args']=['-vf', 'zscale=rin=full:pin=709:tin=linear:min=709:r=full:npl=100:p=2020:t=arib-std-b67:m=2020_ncl,format=yuv444p12le', '-color_range', 'pc', '-color_trc', 'arib-std-b67', '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc', '-x265-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full:max-cll=100,100']
			#replacements['hdr_args']=['-vf', 'zscale=tin=linear:t=arib-std-b67', '-color_range', 'pc', '-color_trc', 'arib-std-b67', '-color_primaries', 'bt709', '-colorspace', 'bt709']
			#replacements['hdr_args']=['-vf', 'zscale=tin=linear:pin=709:min=709:t=arib-std-b67:npl=203:m=bt2020nc,format=yuv444p12le', '-color_range', 'pc', '-color_trc', 'arib-std-b67', '-color_primaries', 'bt709', '-colorspace', 'bt709', '-x265-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,203']
			replacements['hdr_args']=['-vf', 'colorspace=bt2020:ispace=bt709:itrc=bt709:iprimaries=bt2020:trc=bt2020-12:irange=pc:range=pc:format=yuv444p12', '-color_range', 'pc', '-color_trc', 'arib-std-b67', '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc', '-x265-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=400']
		else:
			replacements['hdr_args'] = None
		if self.renderer.options.render_video_args:
			replacements['extra_args'] = shlex.split(self.renderer.options.render_video_args)
		else:
			replacements['extra_args'] = []
		cmd = build_shell_command(FFMPEG_CMDLINE, replacements)
		self.dithering = renderer.options.dithering
		self.ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, pass_fds=list(i.r for i in audio_inputs))
		self.ffmpeg_inputs = {
			'sound': audio_inputs,
			'video': [self.ffmpeg.stdin],
		}

		self.video_framerate_controller = FrameRateController(self.renderer.options.fps, self.renderer.options.render_video_fps)
		self.current_frame = array.array('f', [0] * self.renderer.options.w * self.renderer.options.h * 3)
		self.planar_frame = array.array('f', [0] * self.renderer.options.w * self.renderer.options.h * 3)
		self.motion_blur = self.video_framerate_controller.out_fps < self.video_framerate_controller.fps
		self.framebuffer = None
		self.image = None
		self.shader = None
		self.frame = None
		self.current_frame_number = 0
		self.output_frame_number = 1
		if self.motion_blur or self.dithering:
			self.framebuffer = gl.glGenFramebuffers(1)
			self.image = gl.glGenTextures(1)

			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.image)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.renderer.options.w, self.renderer.options.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_SHORT, None)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
			gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.image, 0)
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
			gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

			self.shader = Shader(
				self.renderer.process_shader(TEXTURE_VERTEX_SHADER),
				self.renderer.process_shader(VIDEO_FRAGMENT_SHADER)
			)
			self.shader.use()
			gl.glUniform1i(self.shader.get_uniform("dithering"), self.dithering)

	def render(self):
		frame_action = self.video_framerate_controller.on_frame()

		if self.renderer.sound:
			while not self.ffmpeg_inputs['sound'][0].w_full() and not self.renderer.sound.buffers.empty():
				self.ffmpeg_inputs['sound'][0].write(self.renderer.sound.buffers.get().tobytes())

		if self.motion_blur or self.dithering:
			self.shader.use()
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)

			gl.glActiveTexture(gl.GL_TEXTURE0)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.renderer.output.image)
			gl.glUniform1i(self.shader.get_uniform("input_buffer"), 0)
			gl.glActiveTexture(gl.GL_TEXTURE1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.image)
			gl.glUniform1i(self.shader.get_uniform("output_buffer"), 1)
			gl.glUniform1i(self.shader.get_uniform("current_frame"), self.current_frame_number)
			gl.glUniform1i(self.shader.get_uniform("output_frame_number"), self.output_frame_number)
			gl.glUniform1i(self.shader.get_uniform("average_frames"), (self.current_frame_number if frame_action.emit_frames else 0) + 1)

			gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.renderer.vertex_surface_buffer)
			self.renderer.enable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
			gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
			self.renderer.disable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
			gl.glFinish()
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

		if frame_action.emit_frames:
			self.current_frame_number = 0
			framebuffer = self.framebuffer if self.motion_blur or self.dithering else self.renderer.output.framebuffer
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
			gl.glReadPixels(0, 0, self.renderer.options.w, self.renderer.options.h, gl.GL_RGB, gl.GL_FLOAT, self.current_frame.buffer_info()[0])
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
			self.vflip_current_frame()
			plane_size=len(self.planar_frame)//3
			self.planar_frame[plane_size*0:plane_size*1] = self.current_frame[1::3]
			self.planar_frame[plane_size*1:plane_size*2] = self.current_frame[2::3]
			self.planar_frame[plane_size*2:plane_size*3] = self.current_frame[0::3]
			for __ in range(frame_action.emit_frames):
				self.ffmpeg_inputs['video'][0].write(self.planar_frame.tobytes())
			self.output_frame_number += 1
		else:
			self.current_frame_number += 1

	def vflip_current_frame(self):
		row_width = self.renderer.options.w * 3
		for row in range(self.renderer.options.h // 2):
			start = row * row_width
			end = start + row_width
			self.current_frame[start:end], self.current_frame[-end:None if start == 0 else -start] = self.current_frame[-end:None if start == 0 else -start], self.current_frame[start:end]

	def destroy(self):
		if self.ffmpeg is not None:
			for input_list in self.ffmpeg_inputs.values():
				for input_stream in input_list:
					input_stream.close()
			self.ffmpeg.wait()


class BufferRenderPass(ImageRenderPass):
	_framebuffer_internal_format = gl.GL_RGBA32F
	_fragment_shader_template = BUFFER_FRAGMENT_SHADER_TEMPLATE


class SoundRenderPass(RenderPass):
	_fragment_shader_template = SOUND_FRAGMENT_SHADER_TEMPLATE
	sample_size = AUDIO_FRAME_SIZE
	sample_rate = 44100
	current_sample = 0
	buffers = queue.Queue(maxsize=2)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.framebuffer = gl.glGenFramebuffers(1)
		self.sample = gl.glGenTextures(1)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.sample)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.sample_size, self.sample_size, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
		gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.sample, 0)
		gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

		gl.glUniform1f(self.shader.get_uniform("iSampleRate"), self.sample_rate)

	def get_fragment_shader_replacements(self):
		replacements = super().get_fragment_shader_replacements()
		replacements['inputs'] = replacements['inputs'] + "uniform float     iTimeOffset;\nuniform int       iSampleOffset;\n"
		return replacements

	def render(self):
		if not self.buffers.full():
			self.shader.use()
			self.bind_inputs()

		while not self.buffers.full():
			gl.glUniform1f(self.shader.get_uniform("iTimeOffset"), self.current_sample / self.sample_rate)
			gl.glUniform1i(self.shader.get_uniform("iSampleOffset"), self.current_sample)
			gl.glViewport(0, 0, self.sample_size, self.sample_size)
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)
			gl.glActiveTexture(gl.GL_TEXTURE0)
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.sample)
			self.renderer.enable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
			gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
			self.renderer.disable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

			gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)
			buf = array.array('B', [0] * self.sample_size * self.sample_size * 4)
			gl.glReadPixels(0, 0, self.sample_size, self.sample_size, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf.buffer_info()[0])
			self.buffers.put(buf)
			self.current_sample += AUDIO_FRAME_SIZE*AUDIO_FRAME_SIZE

	def get_texture(self):
		return Texture(gl.GL_TEXTURE_2D, self.sample)

	def get_framebuffer(self):
		return self.framebuffer


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
		self.render_video_codec = args.render_video_codec
		self.render_video_preset = args.render_video_preset
		self.render_video_crf = args.render_video_crf
		self.render_video_pix_fmt = args.render_video_pix_fmt
		self.render_video_hdr = not args.no_render_video_hdr
		self.render_video_args = args.render_video_args
		self.benchmark = args.benchmark
		if (args.render_video or self.benchmark) and self.fps is None:
			self.fps = 60
		if args.render_video and self.render_video_fps is None:
			self.render_video_fps = self.fps
		self.quiet = args.quiet
		self.no_window = args.no_window
		self.antialias = args.antialias
		self.shutter_speed = args.shutter_speed
		self.dithering = args.dithering
		self.glsl_version = args.glsl_version
		self.max_duration = args.max_duration


class Renderer(object):
	def __init__(self, shader_definition, options):
		self.options = options
		self.vertex_surface_data = array.array('f', [-1,+1,+1,+1,-1,-1,+1,-1])
		self.vertex_surface_buffer = gl.glGenBuffers(1)
		self.vertex_surface_array = gl.glGenVertexArrays(1)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertex_surface_data.itemsize * len(self.vertex_surface_data), self.vertex_surface_data.tobytes(), gl.GL_DYNAMIC_DRAW)

		self.shader = Shader(
			self.process_shader(TEXTURE_VERTEX_SHADER),
			self.process_shader(TEXTURE_FRAGMENT_SHADER)
		)

		self.shader.use()
		#gl.glBindVertexArray(gl.glGenVertexArrays(1))
		#gl.glEnableVertexAttribArray(self.shader.get_attribute("position"))
		#gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		#gl.glVertexAttribPointer(self.shader.get_attribute("position"), 2, gl.GL_FLOAT, False, self.vertex_surface_data.itemsize * 2, ctypes.c_void_p(0))
		#gl.glDisableVertexAttribArray(self.shader.get_attribute("position"))

		self.image_pack_buffer = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.image_pack_buffer)
		gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self.options.pixels_count * 3, None, gl.GL_STREAM_READ)
		gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

		self.tiles = self.__calc_tiles()
		self.render_passes = []
		self.render_passes_by_id = {}
		self.output = None
		self.sound = None
		self.common = ''
		self.start_time = time.monotonic()
		self.real_start_time = datetime.datetime.now()
		self.current_time = self.start_time
		self.last_time = self.start_time
		self.exact_time = fractions.Fraction(0)
		self.frame_durations = []
		self.mouse_state = [0, 0, 0, 0]
		self.mouse_pressed = False
		self.pressed_keys = set()
		self.frame_start = time.monotonic()
		self.shutdown = False

		for render_pass_definition in shader_definition['renderpass']:
			if render_pass_definition['type'] == 'common':
				self.common = render_pass_definition['code']
				if self.common.startswith('file://'):
					self.common = load_from_file(self.common[len('file://'):], self.options.shader_filename, binary=False)

		for render_pass_definition in shader_definition['renderpass']:
			if render_pass_definition['type'] == 'common':
				continue
			render_pass = RenderPass.create(self, render_pass_definition)
			if render_pass.type == 'image':
				self.output = render_pass
			elif render_pass.type == 'sound':
				self.sound = render_pass
			self.render_passes.append(render_pass)
			self.render_passes_by_id[render_pass.output_id] = render_pass
		if self.options.render_video:
			render_pass = RenderPass.create(self, {'type': 'video'})
			self.render_passes.append(render_pass)

		if self.output is None:
			raise RuntimeError("Output is not defined")

		self.frame = 0

	def display(self):
		self.render_offscreen()
		self.render_display()

	def render_offscreen(self):
		if self.options.max_duration is not None and self.exact_time > self.options.max_duration:
			raise SystemExit(0)
		if self.shutdown:
			raise SystemExit(0)
		try:
			gl.glViewport(0, 0, self.options.w, self.options.h)
			self.render_frame()
		except Exception:
			logger.exception("Exception in display method")
			sys.exit()

	def render_display(self):
		gl.glViewport(0, 0, glut.glutGet(glut.GLUT_WINDOW_WIDTH), glut.glutGet(glut.GLUT_WINDOW_HEIGHT))
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		self.shader.use()
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)

		gl.glActiveTexture(gl.GL_TEXTURE0)
		gl.glBindTexture(*self.output.get_texture())
		gl.glUniform1i(self.shader.get_uniform("texture"), 0)

		self.enable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
		gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
		self.disable_surface_vertex_attrib_array(self.shader.get_attribute('position'))
		gl.glFinish()
		glut.glutSwapBuffers()

	def idle(self):
		glut.glutPostRedisplay()

	def reshape(self, width, height):
		gl.glViewport(0, 0, width, height)

	def keyboard_down(self, key, x, y):
		if key == b'\x1b':
			self.quit()
			self.shutdown = True
		self.pressed_keys.add(key)
		for rp in self.render_passes:
			for rp_input in rp.inputs:
				if isinstance(rp_input, KeyboardInput):
					rp_input.keyboard(key, True)

	def keyboard_up(self, key, x, y):
		self.pressed_keys.discard(key)
		for rp in self.render_passes:
			for rp_input in rp.inputs:
				if isinstance(rp_input, KeyboardInput):
					rp_input.keyboard(key, False)

	def keyboard_special(self, key, x, y):
		if key == glut.GLUT_KEY_F12:
			print("Screenshot")

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
			self.exact_time = fractions.Fraction(self.frame, self.options.fps)
		else:
			uniforms['time'] = self.current_time - self.start_time
			uniforms['time_delta'] = self.current_time - self.last_time
			self.exact_time = fractions.Fraction(self.current_time - self.start_time)
		uniforms['frame'] = self.frame
		uniforms['mouse'] = self.mouse_state
		real_date = self.real_start_time + datetime.timedelta(seconds=float(self.exact_time))
		uniforms['date'] = [
			float(real_date.year),
			float(real_date.month - 1),
			float(real_date.day),
			float(real_date.hour * 60 * 60 + real_date.minute * 60 + real_date.second) + float(real_date.microsecond) / 1000000,
		]

		for render_pass in self.render_passes:
			uniforms['channel_time'] = [channel.get_channel_time() for channel in render_pass.inputs]
			uniforms['channel_resolution'] = [channel.get_resolution() for channel in render_pass.inputs]
			render_pass.update_uniforms(uniforms)

		for render_pass in self.render_passes:
			if not isinstance(render_pass, VideoRenderPass):
				render_pass.render()
		gl.glFinish()
		frame_end = time.monotonic()
		self.frame_durations = self.frame_durations[-FRAME_DURATION_AVERAGE+1:] + [frame_end - self.frame_start]
		self.frame_start = frame_end

		for render_pass in self.render_passes:
			if isinstance(render_pass, VideoRenderPass):
				render_pass.render()

		fps = 1.0 / statistics.mean(self.frame_durations)
		fps_text = f'{fps:8.8}'
		hour = int(uniforms['time']) // 3600
		minute = (int(uniforms['time']) - hour * 3600) // 60
		second = (uniforms['time'] - hour * 3600 - minute * 60)

		if not self.options.quiet:
			sys.stdout.write(f"\x1b[2K\rf: {self.frame:<5} fps: {fps_text:<10} time: {hour}:{minute:02}:{int(second):02}.{int(second*100%100):02}")
			sys.stdout.flush()

		self.frame += 1
		self.last_time = self.current_time
		self.current_time = time.monotonic()

	def quit(self):
		for render_pass in self.render_passes:
			render_pass.destroy()

	def get_render_pass_by_id(self, output_id):
		return self.render_passes_by_id[output_id]

	def enable_surface_vertex_attrib_array(self, attribute):
		gl.glBindVertexArray(self.vertex_surface_array)
		gl.glEnableVertexAttribArray(attribute)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_surface_buffer)
		gl.glVertexAttribPointer(attribute, 2, gl.GL_FLOAT, False, self.vertex_surface_data.itemsize * 2, ctypes.c_void_p(0))

	def disable_surface_vertex_attrib_array(self, attribute):
		gl.glDisableVertexAttribArray(attribute)

	def process_shader(self, shader):
		version_str = '#version ' + self.options.glsl_version
		if ' es' in self.options.glsl_version:
			version_str += '\nprecision highp float;\nprecision highp int;\nprecision mediump sampler3D;\n'
		return shader.replace('#version glsl_version', version_str)

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


def parse_duration(val):
	msg = "Duration in wrong format, expected h:m:s"
	rx_match = re.match(r'^(?:(\d+):)?(?:(\d+):)?(?:(\d+))$', val)
	if rx_match is None:
		raise argparse.ArgumentTypeError(msg)
	seconds = rx_match.group(3)
	minutes = rx_match.group(2)
	hours = rx_match.group(1)
	seconds = int(seconds) if seconds else 0
	minutes = int(minutes) if minutes else 0
	hours = int(hours) if hours else 0
	return hours * 3600 + minutes * 60 + seconds


def parse_shadertoy_url(val):
	val = val.split('?')[0]
	match = re.match(r'https://www.shadertoy.com/view/(\w+)', val)
	if match is None:
		raise argparse.ArgumentTypeError("Wrong URL, expected https://www.shadertoy.com/view/code")
	return match.group(1)


def render(args):
	options = RendererOptions(args)

	if options.no_window:
		import OpenGL.EGL as egl
		display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
		major, minor = egl.EGLint(), egl.EGLint()
		egl.eglInitialize(display, ctypes.pointer(major), ctypes.pointer(minor))
		sys.stdout.write(f"EGL: {major.value}.{minor.value}\n")
		egl_config = egl.EGLConfig()
		num_configs = egl.EGLint()
		def egl_convert_to_dict_array(attrs):
			attrs = sum(([k, v] for k, v in attrs.items()), []) + [egl.EGL_NONE]
			return (egl.EGLint * len(attrs))(*attrs)
		egl_config_attribs = {
			egl.EGL_RED_SIZE: 8,
			egl.EGL_GREEN_SIZE: 8,
			egl.EGL_BLUE_SIZE: 8,
			egl.EGL_ALPHA_SIZE: 8,
			egl.EGL_DEPTH_SIZE: egl.EGL_DONT_CARE,
			egl.EGL_STENCIL_SIZE: egl.EGL_DONT_CARE,
			egl.EGL_RENDERABLE_TYPE: egl.EGL_OPENGL_BIT,
			egl.EGL_SURFACE_TYPE: egl.EGL_DONT_CARE,
		}
		egl_config_attribs = egl_convert_to_dict_array(egl_config_attribs)
		if not egl.eglChooseConfig(display, egl_config_attribs, ctypes.pointer(egl_config), 1, ctypes.pointer(num_configs)) or num_configs.value == 0:
			sys.stderr.write("EGL config not nfound\n")
			sys.exit()
		if not egl.eglBindAPI(egl.EGL_OPENGL_API):
			sys.stderr.write("OpenGL API not bound\n")
			sys.exit()
		egl_config_attribs = {
			egl.EGL_CONTEXT_MAJOR_VERSION: 3,
			egl.EGL_CONTEXT_MINOR_VERSION: 3,
			egl.EGL_CONTEXT_OPENGL_PROFILE_MASK: egl. EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
		}
		egl_config_attribs = egl_convert_to_dict_array(egl_config_attribs)
		context = egl.eglCreateContext(display, egl_config, egl.EGL_NO_CONTEXT, egl_config_attribs)
		if not context:
			sys.stderr.write("EGL context not created\n")
			sys.exit()
		if not egl.eglMakeCurrent(display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, context):
			sys.stderr.write("eglMakeCurrent call failed\n")
			sys.exit()

		renderer = Renderer(json.load(options.file), options)
		while True:
			try:
				renderer.render_offscreen()
			except (KeyboardInterrupt, SystemExit):
				sys.stdout.write("Stopping\n")
				sys.stdout.flush()
				renderer.quit()
				sys.exit()
	else:
		os.environ['vblank_mode'] = '0'

		glut.glutInit()
		glut.glutInitContextVersion(3, 3)
		glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE)
		glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
		glut.glutInitWindowSize(*options.resolution)
		win = glut.glutCreateWindow("Shadertoy")
		if options.no_window:
			glut.glutDestroyWindow(win)
		else:
			glut.glutReshapeWindow(*options.resolution)

		renderer = Renderer(json.load(options.file), options)
		glut.glutDisplayFunc(renderer.render_display)
		#glut.glutReshapeFunc(renderer.reshape)
		glut.glutMouseFunc(renderer.mouse)
		glut.glutMotionFunc(renderer.mouse_motion)
		glut.glutKeyboardFunc(renderer.keyboard_down)
		glut.glutKeyboardUpFunc(renderer.keyboard_up)
		glut.glutSpecialFunc(renderer.keyboard_special)
		#glut.glutIdleFunc(renderer.idle)

		while True:
			try:
				glut.glutMainLoopEvent()
				renderer.render_offscreen()
				glut.glutMainLoopEvent()
				if not options.no_window:
					glut.glutPostRedisplay()
			except (KeyboardInterrupt, SystemExit):
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
			if not basename:
				basename = renderpass['type']
			existing_names.add(basename)
			filename = to_local_filename(basename + '.frag', args.file.name)
			with open(filename, 'w') as fp:
				fp.write(renderpass['code'])
			renderpass['code'] = 'file://' + basename + '.frag'
		with open(tmp_filename, 'w') as fp:
			json.dump(shader, fp, indent='\t')
		os.rename(tmp_filename, args.file.name)
	except Exception:
		try:
			os.remove(tmp_filename)
		except FileNotFoundError:
			pass
		raise


def download(args):
	data = {'s': json.dumps({'shaders': [args.url]})}
	data = urllib.parse.urlencode(data).encode('ascii')
	req = urllib.request.Request("https://www.shadertoy.com/shadertoy", data=data)
	req.add_header('Referer', 'https://www.shadertoy.com/view/' + args.url)
	req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0')
	with urllib.request.urlopen(req) as f:
		data = f.read().decode('utf-8')
		json_data = json.loads(data)
		if not json_data:
			raise RuntimeError("Shader not found")
		dirname = os.path.dirname(args.file)
		if dirname:
			os.makedirs(dirname, exist_ok=True)
		with open(args.file, 'w') as fp:
			json.dump(json_data[0], fp, indent='\t')


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
	parser_render.add_argument('--render-video-codec', type=str, choices=['h264', 'h265'], default='h265', help="Video codec")
	parser_render.add_argument('--render-video-preset', type=str, default='medium', help="FFMpeg preset settings")
	parser_render.add_argument('--render-video-crf', type=int, default=20, help="Constant rate factor, less is better")
	parser_render.add_argument('--render-video-pix-fmt', type=str, default='yuv444p12le', help="Pix fmt")
	parser_render.add_argument('--no-render-video-hdr', action='store_true', help="Don't render HDR")
	parser_render.add_argument('--render-video-args', type=str, help="Extra arguments for ffmpeg")
	parser_render.add_argument('--benchmark', action='store_true', help="Benchmark")
	parser_render.add_argument('--quiet', action='store_true', help="Queit")
	parser_render.add_argument('--no-window', action='store_true', help="No window")
	parser_render.add_argument('--antialias', type=parse_resolution, default=(1, 1), help="Antialiasing")
	parser_render.add_argument('--shutter-speed', type=float, default=1.0, help="Shutter speed (only if antialias is enabled)")
	parser_render.add_argument('--dithering', type=int, default=0, help="Enable dithering")
	parser_render.add_argument('--glsl-version', type=str, default='330', help="OpenGL shading language version")
	parser_render.add_argument('--max-duration', type=parse_duration, help="Maximum video duration (HH:MM:SS)")

	parser_extract_sources = subparsers.add_parser('extract_sources', help="Extract shader sources")
	parser_extract_sources.add_argument('file', type=argparse.FileType('r'), help="Shader file")

	parser_extract_sources = subparsers.add_parser('download', help="Download shader from shadertoy.com")
	parser_extract_sources.add_argument('url', type=parse_shadertoy_url, help="Shadertoy URL")
	parser_extract_sources.add_argument('file', type=str, help="Output filename")

	args = parser.parse_args()

	globals()[args.command](args)


if __name__ == "__main__":
	main()
