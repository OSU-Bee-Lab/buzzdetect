Tuning buzzdetect settings for maximum performance
====================================================

There are a few settings available to tweak that can significantly impact analysis rate.
buzzdetect defaults reasonable defaults, but the optimal values will significantly vary between machines.
Try out some different options to see what works best for your analyses.
Tuning your settings may take some time, but across large datasets the performance gains can add up to hours!


A synopsis of the buzzdetect engine pipeline
---------------------------------------------

Here, we'll briefly describe how buzzdetect processes audio data so you can better understand what impact the different settings have.
Settings that can be adjusted are given in bold, with names as they appear in the app.

Streamers
^^^^^^^^^^

1. At the start of analysis, the files in the audio directory are scanned and compared to the results in the output directory, if any exist.
   Any un-analyzed files or incomplete files are placed on a queue for analysis.
2. A number of **concurrent streamers** are launched. Each streamer takes an assignment from the queue.
3. The streamer begins reading audio data from the start of the file (or it picks up where an interrupted analysis left off).
4. Once the streamer has read audio data up to the **chunk length**, it places that data into a queue for the analyzer to process, then starts on the next chunk.
5. The streamer continues reading audio data until the output queue is full;
   the queue can hold a number of chunks equal to its **stream buffer depth**.
   If a streamer goes to enqueue a chunk and the queue is full, it waits until the analyzer(s) take a chunk out of the queue for processing.
   Multiple streamers might wait on the queue at the same time.


Analyzers
^^^^^^^^^^

1. A number of **GPU analyzers** and/or **CPU analyzers** are launched.
2. Each analyzer waits for an audio chunk to be land in the queue.
3. On receiving the chunk, the analyzer applies the corresponding model to the audio, producing the neuron activations that buzzdetect ultimately outputs.
   On an Apple Silicon macOS, the model can be run at **reduced precision** for faster analysis (see :doc:`gpu`).
4. The analyzer hands the neuron activations to a results writer for writing to the ``_buzzpart.csv`` file.


Settings to tweak
------------------
Concurrent streamers
^^^^^^^^^^^^^^^^^^^^^
This is probably the place to start, especially if you're using a graphics card.
buzzdetect tries to guess at a reasonable number of streamers, but this is a highly contextual decision.
Err high on this number; an analyzer waiting for audio can tank your analysis rate, but
a streamer waiting to hand off its audio is virtually cost free.
We find good results with as many as 24 streamers to our 1 GPU.

More streamers means more processing is being spent on putting chunks into the queue.
Usually, the streamers outpace the analyzers.
If you have a lot of streamers, they're theoretically drawing processing power away from the analyzers.
However, most of them will simply be waiting for space in the queue, which does not consume resources.
Generally, this is the first knob to try dialing up.

Note that each file gets one streamer, so there's no benefit in launching more streamers than there are files.

You might want more streamers if:
* You're using a GPU
* Your chunks are long
* You're using compressed audio (e.g. MP3s) that need to be decoded, slowing down streaming
* You have many small recordings
* You're reading from slow storage (e.g., a hard disk or external storage)
* And in general, any time you're seeing BUFFER BOTTLENECKs reported by analyzers in the logs



Chunk length
^^^^^^^^^^^^^^^

At the GPU level, the most efficient chunk length should be the one that fills the VRAM, but this is also contextual
and depends on trade-offs between system resources.
We find that on a GTX 1650 (4GB VRAM, ~2.5 GB free for analysis) we can pass ~1,200 seconds of audio through YAMNet before running out of VRAM.
However, the fastest chunk length on our machine is ~200 seconds!
We found a similar result for CPU, where an M1 MacBook shows best performance around 200 seconds.
This is probably due to the infamous Python GIL, which prevents workers from being run fully in parallel.


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
The greater benefit comes from launching more streamers, which also creates an effectively larger queue.

Imagine a queue depth of 1 and 10 streamers that are greatly outpacing the analyers.
In this case, at equilibrium, the queue would have 1 chunk in it, but each streamer would be holding a chunk as well, waiting to enqueue it.
In this case, the queue has a functional depth of 11.
Starving the queue would require all 10 streamers to hit hiccups (maybe finishing files at the same time) long enough that the analyzer could completely drain the queue before the streamers picked up again.


GPU Analyzers
^^^^^^^^^^^^^^

buzzdetect is fast on any machine, but GPUs are blazingly fast.
Even a very cheap GPU with CUDA capability can greatly accelerate analysis.
See :doc:`gpu` for which builds support which cards and how to set one up.

Initially, we only allowed a single GPU analyzer to run.
We found that using multiple GPU analyzers can modestly increase analysis rate (+~10%).
This is a little surprising, as we expected the GPU to only be used by one call at a time,
but it may also be in part because the GPU analyzer still has some CPU-bound operations to complete,
so using multiple analyzers keeps the GPU better fed.

We are still investigating the impact of multiple GPU analyzers.
Try tweaking this yourself and see what happens!


CPU Analyzers
^^^^^^^^^^^^^^^

If you are using a GPU, you almost certainly do not want any CPU analyzers.
The optimal strategy is to keep the GPU fed; CPU analyzers contend with the GPU analyzer for chunks and
bog down the CPU operations required to get chunks loaded onto the GPU.

One CPU analyzer should efficiently use all cores available on your machine.
Additional analyzers are only expected to produce marignal benefit.

