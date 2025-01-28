import re

from buzzcode.utils import setthreads

setthreads(1)
# TODO: enable GPU!
# TODO: there are memory issues when running multiple embedding processes (appears to be leak, but might just be high usage),
# maybe the most efficient method is to set many threads, then have only one embedder (esp. if using GPU)


def overlaps(range_frame, range_label, minimum_overlap):
    range_overlap = (range_frame[0] + minimum_overlap, range_frame[1] - minimum_overlap)

    # if label ends before frame starts or label starts after frame ends, False
    if (range_label[1] < range_overlap[0]) or (range_label[0] > range_overlap[1]):
        return False
    else:
        # otherwise, it must be overlapping, no matter how!
        return True


def overlapping_elements(range_index, range_table, minimum_overlap, elements=None):
    """given an input range to index with, what elements of the table (list) match?"""
    ''' if elements is provided, returns matching elements. Else returns a boolean list of matches'''

    overlap_bool = [overlaps(range_index, r_table, minimum_overlap) for r_table in range_table]
    if elements is None:
        return overlap_bool

    return elements[overlap_bool]


def expand_chunk(chunk, framelength, event_overlap_s, audio_duration):
    start = max(
        chunk[0] - ((framelength - event_overlap_s) * 0.99),
        # nudge start a bit forward, rounding can lead to accidental nulls where real overlap is *just* less than tolerable
        0  # if start < 0, round up to 0
    )

    end = min(
        chunk[1] + (framelength - event_overlap_s),
        audio_duration  # if end > file length, round down to file length
    )

    if (end - start) < framelength:
        print(f'unexpandable sub-frame audio at; returning null')
        return

    return start, end


def clean_name(name_in, prefix, extension):
    name_out = re.sub(prefix, '', name_in)
    name_out = re.sub(extension, '', name_out)

    return name_out

