Tuning buzzdetect settings for maximum performance
====================================================

There are a few settings available to tweak when running buzzdetect analyses.
Some of these settings are irrelevant to performance, e.g. which neurons to output, what items to log, and where the results are stored.
Other settings significantly impact the analysis rate.
buzzdetect defaults to what we believe are reasonable settings, but we are only able to test them on a limited number of machines.
Try out some different options to see what works best for your analyses.
Tuning your settings may take some time, but across large datasets the performance gains can be significant.


A synopsis of the buzzdetect pipeline
---------------------------------------

Here, we'll briefly describe how buzzdetect processes audio data so you can better understand what impact the different settings have.
Settings that can be adjusted are given in bold, with names as they appear in the buzzdetect GUI.

Streamers
^^^^^^^^^^

1. A number of **concurrent streamers** are launched. Each streamer is assigned its own audio file.
2. The streamer begins reading audio data from the start of the file (or it picks up where an interrupted analysis left off).
3. Once the streamer has read audio data up to its set **chunk length**, it places that data into a queue for the analyzer to process.
4. The streamer continues reading audio data until the queue is full;
   the queue can hold a number of chunks equal to its **stream buffer depth**.
   If a streamer goes to enqueue a chunk and the buffer is full, it waits until a chunk is taken out of the queue by the analyzer.
   Multiple streamers might wait on the queue at the same time.


Analyzers
^^^^^^^^^^

1. A number of **GPU analyzers** and **CPU analyzers** are launched.
2. Each analyzer waits for an audio chunk to be streamed into the queue.
3. On receiving the chunk, the analyzer applies an embedder model.
   This is a pre-trained model, not one produced by the OSU Bee Lab but one that we can leverage to fine-tune our own models.
   The embedder outputs embeddings, lower-dimension, higher-information-density representations of our input audio for prediction.
   The embedder model has some **frame hop** that determines the spacing between frames.
   The frame hop is given as a proportion of the frame length, where a frame is the discrete duration of audio over which a prediction is made.
   For example, YAMNet has a frame length of 0.96s.
   Giving YAMNet a frame hop of 1.0 means that the second frame will come 0.96s after the first, producing contiguous frames.
4. The analyzer then applies the prediction model, which has been fine-tuned on our classes of interest.
   This produces the neuron activations that buzzdetect ultimately outputs.
5. The model hands output the neuron activations to a results writer.


Settings to tweak
------------------
Concurrent streamers
^^^^^^^^^^^^^^^^^^^^^

More streamers means more processing is being spent on putting chunks into the queue.
If resources are limited, it theoretically means less processing spent on analysis.
However, if the streamers are outpacing the analyzer, most of them are simply waiting to enqueue, which does not consume resources.

buzzdetect tries to guess at a reasonable number of streamers, but this is a highly contextual decision.
You might want more streamers if:
* You're using a GPU
* Your chunks are long
* You're using compressed audio (e.g. MP3s) that need to be decoded, slowing down streaming
* You're reading from slow storage (e.g., a hard disk or external storage)
* And in general, any time you're seeing BUFFER BOTTLENECKs reported by analyzers in the logs

Note that each file gets one streamer.
There's no point in launching more streamers than you have audio files, any surplus streamers will immediately shut themselves down.


Chunk length
^^^^^^^^^^^^^^^

At the GPU level, the most efficient chunk length is the one that fills the VRAM.
Looking at the full pipeline, this turns out to be far from the case.
We find that on a GTX 1650 (4GB VRAM, ~2.5 GB free for analysis) we can fit ~1,200 seconds of audio into the embedder (embedders are way more memory intensive than the prediction models) YAMNet_k2.
However, the fastest chunk length on our machine is ~200 seconds!
We found a similar result for CPU, where an M1 MacBook shows best performance aruond 200 seconds with model_general_v3.
This is probably due to the infamous Python GIL, which means that we can't ever really run workers fully in parallel.


You might want a shorter chunk length if:

* You're seeing BUFFER BOTTLENECKs reported by analyzers in the logs
* You're using multiple GPU analyzers

You might want a longer chunk length if:

* Your streamers are outpacing your analyzer


Stream buffer depth
^^^^^^^^^^^^^^^^^^^^^

A bigger buffer lets your streamers get further ahead of the analyzer.
If the queue can hold hundreds of chunks,
the streamers only need to be a bit faster than the analyzer to eventually fill the queue and give themselves some breathing room.
We don't find that this setting makes a big difference in practice.
The greater benefit comes from launching more streamers, which creates an effectively larger queue.

Imagine a queue depth of 1 and 10 streamers that are greatly outpacing the analyers.
In this case, at equilibrium, the queue would have 1 chunk in it, but each streamer would be holding a chunk as well, waiting to enqueue it.
In this case, the queue has a functional depth of 11.
Starving the queue would require all 10 streamers to hit hiccups (maybe finishing files at the same time) long enough that the analyzer could completely drain the queue before the streamers picked up again.


GPU Analyzers
^^^^^^^^^^^^^^

buzzdetect is fast on any machine, but GPUs are blazingly fast.
Even a very cheap GPU with CUDA capability can greatly accelerate analysis.

Initially, we only allowed a single GPU analyzer to run.
We found that using multiple GPU analyzers can modestly increase analysis rate (+~10%).
This is a little surprising, as we expected the GPU to only be used by one call at a time,
but it may also be in part because the GPU analyzer still has some CPU-bound operations to complete,
so using multiple analyzers keeps the GPU better fed.

We are still investigating the impact of multiple GPU analyzers, but we only have one machine for testing.
Try tweaking this yourself and see what happens.

Note: multiple analyzers appear to compete for GPU resources.
We found that if we run multiple GPU analyzers at the highest chunk length that fits into our VRAM,
the process throws OOM exceptions.


CPU Analyzers
^^^^^^^^^^^^^^^

If you are using a GPU, you almost certainly do not want any CPU analyzers.
If you are running on CPU only, we find that TensorFlow already efficiently manages multithreading.
You might see the highest rate on a single CPU analyzer, but you can also try using two.

Frame hop
^^^^^^^^^^^

**WARNING**: changing this value will change inference.
A frame hop of 2 skips every other frame.
We also find that (unexpectedly) frame hops less than 1 do not improve the sensitivity of the model at a fixed precision.
Finally, not all models support variable framehops.

Our YAMNet embedding model is more efficient with an older version of Keras, which we have saved as "yamnet_k2".
The downside is that this model does not support variable framehop.

Doubling framehop will increase analysis rate, but not by 2x due to the other operations that are happening during analysis.

We recommend leaving framehop to 1.0 for all analyses.


Eliminating bottlenecks
------------------------
In theory, you want to feed your analyzers at exactly the rate they're consuming chunks.
In reality, because you're seeking to max out your analysis rate above all else,
and because you're not likely to be using your machine to do anything else while analysis is running
(that is, you don't need to minimize compute), you probably want your streamers to be faster than your analyzer.
An analyzer waiting to take items from the queue is decreasing the analysis rate, but
a streamer waiting to place items into the queue is just waiting patiently.
It isn't even a problem to have dozens of streamers waiting to enqueue; they don't burn any processing power while they're waiting.

Try cranking the streamer count way up and see what happens.
We find good results with as many as 24 streamers to our 1 GPU, though your optimal configuration may be different depending on your hardware and audio files.

