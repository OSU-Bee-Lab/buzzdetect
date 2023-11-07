
# change path in to dir in, read files automatically; change chunking dir to processing
def analyze_multithread(modelname, threads, storage_allot, memory_allot, dir_in="./audio_in", dir_out="./output", dir_proc="./processing", chunklength=None):
    model, classes = loadUp(modelname)

    # Processing
    #
    paths_raw = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))



    storage_preprocessing = f
    # Chunking
    #
    storage_allot_chunk = f

    if memory_allot > 16:
        memory_allot = 16
        print("Warning: chosen memory allotment causes overflow errors; memory allotment reduced to 16GB")

    chunk_limit = size_to_runtime(min(memory_allot, storage_allot/threads))
    chunk_limit = chunk_limit/8 # account for expansion in memory
    chunk_limit = chunk_limit/3600 # convert to hours


    if chunklength is None:
        chunklength = chunk_limit
        print("Automatically setting chunk length to maximum: " + str(chunk_limit.__round__(1)) + " hours")
    else:
        if chunklength > chunk_limit:
            chunklength = chunk_limit
            if memory_allot < (storage_allot/threads):
                print("Warning: desired chunk length exceeds memory allotment; reducing chunk length to " + str(chunk_limit.__round__(1)) + " hours")
            else:
                print("Warning: sum of chunks in batch exceeds storage allotment; reducing chunk length to " + str(chunk_limit.__round__(1)) + " hours")



    #
    # Build control ----
    #

    dir_chunk = os.path.join(dir_proc, 'chunks')

    chunkdf_list = []
    for path in paths_raw:
        chunkdf = pd.DataFrame(make_chunklist(path, chunklength=chunklength), columns=["start", "end"])
        chunkdf['path_in'] = path
        chunkdf['chunknum'] = chunkdf.reset_index().index # returns row numbers
        chunkdf_list.append(chunkdf)

    control_chunk = pd.concat(chunkdf_list)

    control_chunk['filetype'] = control_chunk['path_in'].apply(lambda x: os.path.splitext(x)[1]) # should allow analysis with any filetype ffmpeg supports

    def get_path_chunk(path_raw, filetype, start):
        path_chunk = re.sub(pattern=filetype, repl="_s" + str(start) + ".wav", string=path_raw)
        path_chunk = re.sub(pattern=dir_in, repl=dir_chunk, string=path_chunk)
        return path_chunk

    def get_path_out(path_chunk):
        path_out = re.sub(pattern=".wav$", repl="_buzzdetect.txt", string=path_chunk)
        path_out = re.sub(pattern=dir_chunk, repl=dir_out, string=path_out)
        return path_out

    control_chunk['path_chunk'] = control_chunk.apply(lambda x: get_path_chunk(x.path_in, x.filetype, x.start), axis = 1)
    control_chunk['path_out'] = control_chunk['path_chunk'].apply(lambda x: get_path_out(x))

    control_chunk = control_chunk.sort_values(by = ["chunknum", "path_in"])

    get_unique_dirs(control_chunk['path_chunk'])
    get_unique_dirs(control_chunk['path_out'])

    batches = list(range(0, (len(control_chunk)/threads).__ceil__()))

    for batch in batches:
        batch_start = batch * threads
        batch_end = (batch_start + threads) - 1

        control_sub = control_chunk[batch_start:batch_end]

        print("chunking files: \n" + str(control_sub['path_chunk']))
        take_chunks_from_control(control_sub)

        for r in list(range(0, len(control_sub))):
            row = control_sub.iloc[r]

            print("analyzing chunk " + row['path_chunk'])
            analysis = analyze_wav(
                model,
                classes,
                wav_path=row['path_chunk']
            )

            analysis.to_csv(row['path_out'])
            os.remove(row['path_chunk'])

    # delete the processing folder
