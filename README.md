[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.15537954.svg)](https://doi.org/10.5281/zenodo.15537954)

# buzzdetect
<div align="center">
    <img src="docs/source/_images/title_transparent.png" alt="Project Header" />
</div>

buzzdetect is a tool for passive acoustic monitoring of pollinator activity. It uses machine learning to analyze audio recordings and identify the buzz of insect flight, enabling highly scalable, temporally rich observation. Read the peer-reviewed paper in the [Journal of Insect Science](https://doi.org/10.1093/jisesa/ieaf104). 

Read [our full documentation](https://buzzdetect.readthedocs.io/en/latest/) on readthedocs.io. Documentation is still underway; please bear with us!

## Key Features

- **Automated observation.** Enables passive acoustic monitoring of pollinators by detecting the buzz of insect flight in audio. Drop your recorders in the field and let them do your observation for you.

- **Big data.** Support for arbitrarily large datasets. Input audio files can be days long, input datasets can be years long — buzzdetect intelligently streams audio one chunk at a time. Interrupted analyses can pick right back up from where you left off, no data lost!

- **From sounds to stats.** Check out our companion package, [buzzr](https://github.com/OSU-Bee-Lab/buzzr) and [our walkthrough](https://lukehearon.com/blog/2026/buzzdetect-walkthrough/) for everything you need to go from recordings to results.

- **Flexible application.** Support for a wide variety of audio formats; run analyses through command line, Python API, and graphical interface.

- **It's FOSS!** buzzdetect's source code is licensed under MIT, free as in speech, free as in pizza. **Note:** embedding models could be subject to their own licenses — check out [NOTICE](https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/NOTICE) and [LICENSES/](https://github.com/OSU-Bee-Lab/buzzdetect/tree/main/LICENSES) for more info.

## Citing buzzdetect 
If you want to cite buzzdetect in a scholarly work, please cite [the paper](https://doi.org/10.1093/jisesa/ieaf104) for the method; for reproducability, cite and [the Zenodo DOI](https://doi.org/10.5281/zenodo.15537954) corresponding to the version you used in your analysis.  
