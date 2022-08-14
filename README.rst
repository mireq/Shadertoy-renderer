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
Video output
   Shader can be rendered to video file (with HDR support).

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

--resolution                   Window or video size in format ``width`` x ``height``
--tile-size                    Tile size in format ``width`` x ``height``
--fps num                      Target frames per second (valid for video)
--render-video file            Path to video file
--render-video-fps num         If this frame rate is lower than ``fps``, then
                               multiple frames are used to render motion blur.
--render-video-codec codec     Valid values are  ``h264`` or ``h265``
--render-video-preset preset   Attribute ``preset`` passed to ffmpeg.
--render-video-crf num         Attribute ``crf`` passed to ffmpeg (lower value
                               = better quality)
--render-video-pix-fmt fmt     Attribute ``pix_fmt`` passed to ffmpeg. Default
                               value is yuv444p12le (12 bit without chroma
                               subsampling)
--no-render-video-hdr          Don't render to HDR (video is rendered as HDR
                               without this option)
--render-video-args args       Additional arguments passed to ffmpeg
