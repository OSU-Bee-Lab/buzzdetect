# So much to do...
## Preprocessing pipeline
* Do we need to get Reed's snipper code working?
* MP3 → WAV; on supercomputer instead of local?
  - Can we parallelize the process?

## Code
### generatemodel.py
* Store copy of metadata csv within model directory; or is it already in the object? What about training parameters like number of epochs?

### analyzeaudio.py
* Rename to something more useful (maybe just name each script after the action: train, analyze, preprocess)
* Rewrite to input files from any directory structure (e.g., all WAVs within the input path) and output with a cloned directory structure
* Output confidence score as probability?
    - Issue: each frame has its own probability. How do you rigorously turn that into a bee probability?
    - We wouldn't likely even use the probability, except as a cutoff to classify be activity, which we can do with the confidence score anyways.

### buzzdetect.py
* Get new command line operation working

## Machine Learning Design
* Leverage all data by accounting for uneven replication somehow ([adding bias](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) to categories?)

## Supercomputing
### Figure out supercomputing
* This doesn't take long at all to train or analyze a dozen files on my cheap laptop...surely we aren't using the full power of the supercomputer.
  - last I used it, we got a warning about no CUDA cores/TensorRT warnings. The supercomputer has CUDA cores, right? Do we need to make some sort of request when we use them?

### Figure out Teams → OSC file transfer
* Should be able to transfer files directly from Teams (SharePoint Azure Blobs) to the supercomputer. This would be much better in every way.
  - Send as single archive?
  - Send in multiple batches so that processing can start even as more files are arriving?
  - Are we working with terabytes at once?
  - How do you move files from the scratch space to the local storage on the node?
* Could also send files back right to SharePoint?

## Documentation
### README.md
* Is now quite out of date.
* FileZilla information should be updated when we figure out how to integrate SharePoint
* CLI commands and flags will need to change once we update buzzdetect.py
