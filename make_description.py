#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path


def fmt_size(size):
	if size < 8192:
		return f'{size} bytes'
	else:
		size = size / 1024
		size = f'{size:.1f}'.rstrip('0').rstrip('.')
		return f'{size} KiB'


def main():
	parser = argparse.ArgumentParser(description="Add description")
	parser.add_argument('file', type=argparse.FileType('r'), help="Shader file")

	args = parser.parse_args()
	shader = json.load(args.file)
	shader_info = shader['info']
	file_name = Path(args.file.name)
	args.file.close()

	data = {'nt': '1', 'nl': '1', 'np': '1', 's': json.dumps({'shaders': [shader_info['id']]})}
	data = urllib.parse.urlencode(data).encode('ascii')
	req = urllib.request.Request("https://www.shadertoy.com/shadertoy")
	req.add_header('Referer', 'https://www.shadertoy.com/view/' + shader_info['id'])
	try:
		with urllib.request.urlopen(req, data) as f:
			shader = json.loads(f.read().decode('utf-8'))[0]
			shader_info = shader['info']
	except Exception as e:
		sys.stdout.write(str(e) + '\n')

	total_code = ''
	for render_pass in shader['renderpass']:
		code = render_pass['code']
		if code.startswith('file://'):
			with open(file_name.parent / code[len('file://'):], 'r') as fp:
				code = fp.read()
		total_code += code

	code_size = len(total_code.encode('utf-8'))

	shader_name = shader_info['name']
	shader_author = f', author {shader_info["username"]}' if shader_info.get('username') else ''
	shader_tags = ''.join([f', {tag}' for tag in shader_info.get('tags') if tag not in {'shadertoy', 'shader', 'opengl', 'opengl shader'}])
	shader_author = shader_author.replace('"', '\\"')
	shader_tags = shader_tags.replace('"', '\\"')
	shader_size = f', size {fmt_size(code_size)}'

	file_dir = file_name.parent
	dir_name = file_dir.name

	with open(file_dir / 'upload', 'w') as fp:
		fp.write(f'youtube-upload --title "{shader_name} (shadertoy{shader_author}{shader_size})" --tags "shadertoy, shader, opengl, opengl shader{shader_tags}" --description-file description --playlist="Shadertoy" --embeddable=True --privacy unlisted out.mkv\n')

	description = f"Source on shadertoy: https://www.shadertoy.com/view/{shader_info['id']}\nRendered using AMD Ryzen™ 7 PRO 5850U\nSpeed (1080p): {{fps}} fps\nPlaylist: https://www.youtube.com/playlist?list=PLCAFZV4XJzP-jGbTke6Bd3PNDpP1AbIKo\nRenderer: https://github.com/mireq/Shadertoy-renderer\n\n{shader_info['description']}\n"
	with open(file_dir / 'description', 'w') as fp:
		fp.write(description)

	proc = subprocess.Popen(['./shadertoy.py', 'render', str(file_name), '--resolution', '1920x1080',  '--benchmark', '--no-window', '--max-duration', '0:00:02'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	buf = BytesIO()
	while True:
		out = proc.stdout.read(1)
		buf.write(out)
		if out == b'' and proc.poll() != None:
			break
		if out != b'':
			sys.stdout.buffer.write(out)
			sys.stdout.flush()

	buf.seek(0)
	buf = buf.getvalue()
	fps_pos = buf.rfind(b'fps:')
	if fps_pos == -1:
		return
	buf = buf[fps_pos:]
	fps = buf.decode('utf-8').split()[1]
	with open(file_dir / 'description', 'w') as fp:
		fp.write(description.format(fps=fps))


if __name__ == "__main__":
	main()
