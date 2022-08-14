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

   ./shadertoy.py download <url> <file.json>
   # e.g. ./shadertoy.py download https://www.shadertoy.com/view/WsSBzh file.json

Extracting sources
^^^^^^^^^^^^^^^^^^

By default all source files are packed in single json file. To extract sources
run:

.. code:: bash

   ./shadertoy.py extract_sources <file.json>

Now directory contains unpacked source files like:

- common.frag
- image.frag
- buffer_a.frag
- buffer_b.frag
