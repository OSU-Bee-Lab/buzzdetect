::: {align="center"}
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15537954.svg)](https://doi.org/10.5281/zenodo.15537954) [![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://buzzdetect.readthedocs.io/en/latest/) [![Paper](https://img.shields.io/badge/paper-Journal%20of%20Insect%20Science-blue)](https://doi.org/10.1093/jisesa/ieaf104)
:::

# buzzdetect

::: {align="center"}
```         
<img src="docs/source/_images/title_transparent.png" alt="Project Header" />
```
:::

buzzdetect is a tool for passive acoustic monitoring of pollinator activity. It uses machine learning to analyze audio and identify the buzz of insect flight, enabling highly scalable, temporally rich observation. Read the peer-reviewed paper `in the Journal of Insect Science <https://doi.org/10.1093/jisesa/ieaf104>`\_.

## download the app

[![Download for macOS (Apple Silicon)](https://img.shields.io/badge/Download-macOS%20Apple%20Silicon-000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-AppleSilicon.dmg) [![Download for macOS (Intel)](https://img.shields.io/badge/Download-macOS%20Intel-555?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-Intel.dmg) [![Download for Windows](https://img.shields.io/badge/Download-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows.exe) [![Download for Windows (CUDA)](https://img.shields.io/badge/Download-Windows%20CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows-CUDA.zip) [![Download for Linux (.deb)](https://img.shields.io/badge/Download-.deb-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Linux.deb)

See the [full documentation](https://buzzdetect.readthedocs.io/en/latest/getting_started.html) for more information on using the app or underlying Python engine. See our [walkthrough](https://lukehearon.com/blog/2026/buzzdetect-walkthrough/) for a broader overview of conducting a passive acoustic monitoring experiment using buzzdetect.

## buzzdetect at a glance

- **Automate your observations.** Enables passive acoustic monitoring of pollinators by detecting the buzz of insect flight in audio. Drop your recorders in the field and let them do your observation for you.

- **Big ol' data.** Supports arbitrarily large datasets. Input audio files can be days long, input datasets can be years long. buzzdetect will plod through the audio one chunk at a time. And you won't lose your work - interrupted analyses can pick right back up from where you left off, no data lost!

- **Lots of formats.** Support for a wide variety of audio formats - even (the audio part of) some videos! Including: wav, mp3, flac, ogg, aiff, mp4, wma, mts, and a bunch of others.

- **From sounds to stats.** Check out our companion package, `buzzr <https://github.com/OSU-Bee-Lab/buzzr>`\_ and `our walkthrough <https://lukehearon.com/blog/2026/buzzdetect-walkthrough/>`\_ for everything you need to go from recordings to results.

- **It's FOSS!** buzzdetect's source code is licensed under MIT, free as in speech, free as in pizza. Embedding models could be subject to their own licenses. Check out `NOTICE <https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/NOTICE>`\_ and `LICENSES/ <https://github.com/OSU-Bee-Lab/buzzdetect/tree/main/LICENSES>`\_ for more info.

## citing buzzdetect

If you want to cite buzzdetect in a scholarly work, please cite `the paper <https://doi.org/10.1093/jisesa/ieaf104>`\_ for the method; for reproducability, cite and `the Zenodo DOI <https://doi.org/10.5281/zenodo.15537954>`\_ corresponding to the version you used in your analysis.

