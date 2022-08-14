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
- yt-dlp (for downloading audio files)

Features
--------

Tiled rendering
   High resolution rendering can block GPU until the entire frame is rendered.
   This would mean that the computer will be unusable during rendering. Tiled
   rendering can divide work to small tiles and block GPU only for small piece
   of frame.

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

Rendering
^^^^^^^^^

Shader can be rendered using following command:

.. code:: bash

   ./shadertoy.py render <file.json>

Subcommand ``render`` has many optional arguments.

--resolution            Window or video size in format ``width`` x ``height``
--tile-size             Tile size in format ``width`` x ``height``
