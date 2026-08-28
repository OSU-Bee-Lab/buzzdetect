# Step 1: the gate passed, and it moved the goalposts

`TAIL_SCAN_PLAN.md` made Step 1 a gate that could kill the plan: does a decode
started partway into an mp3 converge, exactly, on what a continuous decode
produces? The answer is **yes, if you start on the right frame** — and finding
out which frame that is turned up something the plan did not anticipate, which
is that today's driver does not have one answer to compare against.

Everything below was measured on the `Chia - Solar Eclipse` corpus (8 files,
162.4 audio hours, 48 kbps CBR mono 44.1 kHz, Sony ICD-PX370, no Xing header)
with `engine/.venv`. The scripts named are in this directory.

## The gate itself

**Convergence takes two frames** (2,208 samples, every offset measured), and
after those two frames the fragment either matches the continuous decode
*exactly* or misses it by ~2.4e-7 forever. There is no third outcome and no
decay: the residual is flat over 7.5 million samples.

Which of the two happens **alternates with the start frame**, and the phase is
not a property of the file. `step1_parity.py` walks consecutive boundaries and
finds strict alternation everywhere; `step1_sweep.py` and the diagnostics behind
it find the phase differing between regions of the same file. So no rule
predicts it, and picking a boundary by arithmetic is a coin flip.

Over the whole corpus, decoding **the tail** — the part libsndfile clamps away —
from a boundary N frames before the clamp (`step1_tail.py`):

| overlap | 1249 | 1116 | 1432 | 1125 | 1413 | 1220 | 1247 | 1213 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| even (6, 8, 32, 128, 512) | **exact** | **exact** | 2.4e-7 | **exact** | 1.2e-7 | 2.4e-7 | 1.8e-7 | 2.4e-7 |
| odd (7, 9, 33, 129, 513) | 2.4e-7 | 1.2e-7 | **exact** | 1.2e-7 | **exact** | **exact** | **exact** | **exact** |

Four files want each parity. Depth beyond ~6 frames changes nothing; at 2 frames
the decoder's own warm-up leaks into the comparison, which is why the driver
uses 12. **Exactly one of any two adjacent boundaries is right, always.**

So the design is not to predict the boundary but to measure it: decode a window
the body has already produced through both candidates and keep the one that
comes back identical. That is what `LocalDriver._ensure_tail` does.

## What the gate turned up

Three things, all of which change what "bit-identical" can mean.

**1. A plain open decodes the body exactly as the scanned one does.**
`step1_body.py`: 894,175,796 samples over the whole body, read the way the
streamer reads (seek to each chunk start, one read per chunk, 500 s chunks),
byte for byte identical. This is the half of the plan that had no risk in it,
and it is now measured rather than assumed.

**2. Seeking changes what libsndfile decodes.** `step1_seek.py`: reading a
window after seeking straight to it, versus after seeking to it having already
decoded a second of audio, gives different samples — ~4e5 of 1e6 differing, up
to 1.2e-7. Same file, same offset, same driver, today.

**3. So does breaking a read.** `step1_segmentation.py`: reading 262,144 samples
as one call, versus as two, differs by ~6e-8 unless the break falls exactly on
an MPEG frame boundary. Splits at 1, 1151, 1153, 4096 and 131,072 samples all
perturb it; the split at 1,152 does not.

Both are the same mechanism as the parity above — mpg123's synthesis state ends
up aligned differently and sums the same window in a different order — and both
are properties of the *existing* driver, not of anything added here.

**The consequence for the plan.** Success criterion 1 asked for output
bit-identical to "what today's `LocalDriver` produces for the same file". There
is no such thing: today's driver produces different samples for the same file
depending on where the caller seeks and where it breaks its reads. What is
well defined is the driver's answer *to a given access pattern*, so that is what
the new read path is held to:

- **The streamer's pattern** — `seek(chunk start)`, one `read(chunk length)`,
  which is what `worker.py::queue_chunk` does — must be identical. It is, at
  500 s and at 20 s chunks, over the seam and over the whole file.
- **Any single read crossing the clamp** must equal the old driver's unbroken
  read of the same range. It does.
- **Any read after a seek** must be identical. It is: seeking into the fragment
  was measured to return exactly what seeking to the same place in the whole
  file returns, from either boundary, because a seek resets the decoder.

Two corners are deliberately not identical, both documented in `mp3.py`:

- A caller that **breaks a read at the clamp** and carries on gets the unbroken
  decode from the new driver and the broken one from the old. The new answer is
  the more defensible of the two; they differ by ≤2.4e-7.
- A caller that **breaks reads less than one frame past the clamp** and carries
  on without seeking sees ≤1.5e-8, because the fragment's read boundaries and
  the caller's fall out of step. The streamer never does this: it seeks.

## The other thing that fell out

The scan can go entirely. A CBR stream's frames average exactly
`144 * bitrate / samplerate` bytes — the padding bit is how an integer frame
size tracks a non-integer average — so the audio byte count divided by that
gives the frame count, and `mpg123_scan()`'s whole job collapses into a
division. `step2_arith.py` checks it against the scan: exact on 48 kbps mono and
on 128 kbps stereo, with predicted frame offsets landing within 2 bytes of a
real header across the file. `_read_layout` validates the layout at five points
plus the last frame and refuses the fast path if any of it fails.

## What the corpus actually contains

`survey_corpora.py` over every experiment folder: 4,240 of 4,257 sampled files
are MPEG1 48 kbps 44.1 kHz mono from an ICD-PX370 with no Xing header — the
at-risk shape. Six are 128/192 kbps stereo (`Luke - Audio Fidelity Test`,
`Luke - Wooster Apple`). Eleven are macOS `._` resource forks or 2–4 KB
fragments of dead recordings, which have no MPEG frame at all and fall back.
