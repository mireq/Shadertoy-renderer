==================
Shadertoy renderer
==================

This is advanced shadertoy renderer, downloader and extractor.

.. image:: https://raw.githubusercontent.com/wiki/mireq/Shadertoy-renderer/previews.jpg?v2022-08-14

Sample renders are available in `youtube playlist <https://www.youtube.com/playlist?list=PLCAFZV4XJzP-jGbTke6Bd3PNDpP1AbIKo>`_.

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
Antialiasing
   Antialiasing with diagonal pattern (example ``2x2``, ``3x3``, ``4x4``,
   ``1x2``, ``2x1``, ``2x3``).

   .. image:: https://raw.githubusercontent.com/wiki/mireq/Shadertoy-renderer/aa.png?v2022-08-14
Motion blur
   With enabled antialiasing, pixel samples are redered with slightly shifted
   time, which enables motion blur.
Rendering without GUI
   Running with environment variable ``EGL_PLATFORM=drm`` and option
   ``--no-window`` allows rendering without GUI.

Supported shadertoy features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ☑ Image output
- ☑ Sound output
- ☑ Buffer output
- ☑ Commoon shader code
- ☑ Keyboard input
- ☑ Mouse input
- ☑ Soundcloud input
- ☑ Music input
- ☑ Buffer input
- ☑ Cubemap input
- ☑ Texture input
- ☑ Volume input
- ☐ Cubemap output
- ☐ Webcam input
- ☐ Microphone input
- ☐ Video input

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

--resolution XxY               Window or video size in format ``width`` x ``height``
--tile-size XxY                Tile size in format ``width`` x ``height``
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
--benchmark                    Run without flush commands
--quiet                        Don't show statistics
--no-window                    Run without window
--antialias XxY                Antialiasing with pattern defined as
                               ``x samples`` x ``y samples``
                               This option automatically enables motion blur. To
                               disable motion blur set ``--shutter-speed`` to
                               ``0``
--shutter-speed float          Set shutter speed to fraction of frame duration.
                               Default value is 1.0.
--dithering float              Set dithering intensity
--max-duration time            Max duration of video in format ``HH:MM:SS``

