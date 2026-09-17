buzzdetect: passive acoustic pollinator monitoring
==================================================
.. image:: https://zenodo.org/badge/685544295.svg
   :target: https://doi.org/10.5281/zenodo.15537954

.. image:: https://img.shields.io/github/license/OSU-Bee-Lab/buzzdetect
   :alt: license badge for MIT license

.. figure:: _images/title_transparent.png
    :height: 200px

.. |badge-macos-arm| image:: https://img.shields.io/badge/Download-macOS%20Apple%20Silicon-000?style=for-the-badge&logo=apple&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-AppleSilicon.dmg
   :alt: Download for macOS (Apple Silicon)

.. |badge-macos-intel| image:: https://img.shields.io/badge/Download-macOS%20Intel-555?style=for-the-badge&logo=apple&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-Intel.dmg
   :alt: Download for macOS (Intel)

.. |badge-windows| image:: https://img.shields.io/badge/Download-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows.exe
   :alt: Download for Windows

.. |badge-windows-cuda| image:: https://img.shields.io/badge/Download-Windows%20CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows-CUDA.zip
   :alt: Download for Windows (CUDA)

.. |badge-linux-deb| image:: https://img.shields.io/badge/Download-.deb-FCC624?style=for-the-badge&logo=linux&logoColor=black
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Linux.deb
   :alt: Download for Linux (.deb)

|badge-macos-arm| |badge-macos-intel| |badge-windows| |badge-windows-cuda| |badge-linux-deb|


buzzdetect is a tool for passive acoustic monitoring of pollinator activity.
It uses machine learning to analyze audio recordings and identify the buzz of insect flight, enabling highly scalable, temporally rich observation.
Read the peer-reviewed paper `in the Journal of Insect Science <https://doi.org/10.1093/jisesa/ieaf104>`_.
The paper uses the model ``model_general_v3``; similar tests will be performed on all future models and stored in the model folder.

**Citing buzzdetect**.
If you want to cite buzzdetect in a scholarly work, please cite `the paper <https://doi.org/10.1093/jisesa/ieaf104>`_ for the method;
for reproducability, cite and `the Zenodo DOI <https://doi.org/10.5281/zenodo.15537954>`_ corresponding to the version you used in your analysis.

buzzdetect is under active development, so these docs could drift out of date. Please contact the maintainers if you find an error!

buzzdetect at a glance
-------------

- **Automate your observations.** Enables passive acoustic monitoring of pollinators by detecting the buzz of insect flight in audio.
  Drop your recorders in the field and let them do your observation for you.

- **Big ol' data.** Supports arbitrarily large datasets.
  Input audio files can be days long, input datasets can be years long.
  buzzdetect will plod through the audio one chunk at a time.
  And you won't lose your work - interrupted analyses can pick right back up from where you left off, no data lost!

- **Lots of formats.** Support for a wide variety of audio formats - even (the audio part of) some videos! Including: wav, mp3, flac, ogg, aiff, mp4, wma, mts, and a bunch of others.

- **From sounds to stats.** Check out our companion package, `buzzr <https://github.com/OSU-Bee-Lab/buzzr>`_ and `our walkthrough <https://lukehearon.com/blog/2026/buzzdetect-walkthrough/>`_
  for everything you need to go from recordings to results.

- **It's FOSS!** buzzdetect's source code is licensed under MIT, free as in speech, free as in pizza.
  Embedding models could be subject to their own licenses. Check out `NOTICE <https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/NOTICE>`_ and `LICENSES/ <https://github.com/OSU-Bee-Lab/buzzdetect/tree/main/LICENSES>`_ for more info.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   gpu
   gui
   gui_analysis
   result_files
   tuning

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`