buzzdetect: automated pollinator monitoring
==================================================
.. image:: https://zenodo.org/badge/685544295.svg
   :target: https://doi.org/10.5281/zenodo.15537954

.. image:: https://img.shields.io/github/license/OSU-Bee-Lab/buzzdetect
   :alt: license badge for MIT license

.. figure:: _images/title_transparent.png
    :height: 200px


buzzdetect is a tool for passive acoustic monitoring of pollinator activity.
It uses machine learning to analyze audio recordings and identify the buzz of insect flight, enabling highly scalable, temporally rich observation.
Read the peer-reviewed paper `in the Journal of Insect Science <https://doi.org/10.1093/jisesa/ieaf104>`_.
The paper uses the model ``model_general_v3``; similar tests will be performed on all future models and stored in the model folder.

**Citing buzzdetect**.
If you want to cite buzzdetect in a scholarly work, please cite `the paper <https://doi.org/10.1093/jisesa/ieaf104>`_ for the method;
for reproducability, cite and `the Zenodo DOI <https://doi.org/10.5281/zenodo.15537954>`_ corresponding to the version you used in your analysis.

Documentation is still underway; please bear with us!

Key Features
-------------

* Automated detection of insect buzzes in audio recordings
* Support for arbitrarily large datasets
* Support for multiple audio formats
* Flexible usage through command line, Python API, and graphical interface

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   gui
   workflow

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   cli
   dictionary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`