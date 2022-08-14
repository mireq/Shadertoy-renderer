==================
Shadertoy renderer
==================

This is advanced shadertoy renderer, download and extractor.

Dependencies
------------

Mandatory:

- PyOpenGL

Optional:

- ffmpeg / ffprobe binary (for image and video support)
- numpy (for audio analzyzer support)

Usage
-----

Downloading
^^^^^^^^^^^

To download shader run:

.. code:: bash
	./shadertoy.py download url file.json
	# e.g. ./shadertoy.py download https://www.shadertoy.com/view/WsSBzh shader.json
