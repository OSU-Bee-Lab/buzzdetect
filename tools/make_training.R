#
# Setup ----
#

  library(dplyr)
  library(stringr)
  library(parallel)
  
  project_root = "./localData"
  
  threads <- 7
  
  # This pipeline is split between two steps: snip-generating and wav-converting
  # This may seem like an inefficient approach, and I should just snip and output a wav in one step,
  # but for whatever reason, ffmpeg has been taking extremely long when doing both operations in one step.
  # My suspicion is that ffmpeg is converting a large portion of the file to wav first, then snipping.
  # It's been taking about 1 minute to snip-and-convert 1 file. It's much faster to snip first (~2.6 thread-seconds/file),
  # then convert (negligible/file)
  
#
# Master ----
#
  
  annotations <- data.frame(
    path_annotation = list.files(path = project_root, recursive = T, full.names = T, pattern = "combinedAnnotations")
  ) %>% 
    mutate(
      path_raw = path_annotation %>% 
        str_replace("combined annotations", "raw audio") %>% 
        str_replace(fixed("_combinedAnnotations.txt"), ".mp3"),
      
      name_raw = path_raw  %>% 
        str_split("/", simplify = T) %>% 
        .[,ncol(.)],
      
      mp3_dir = path_annotation %>% 
        str_replace("combined annotations", "mp3 snips") %>% 
        # lazy way to kill the basenames off the end of the path
        str_split("/", simplify = F) %>% 
        lapply(function(str){str[1:(length(str)-1)]}) %>% 
        sapply(function(str) paste0(str, collapse = "/")),
      
      wav_dir = mp3_dir %>% 
        str_replace("mp3", "wav")
    )
  
  ffcommands <- sapply(
    simplify = F,
    1:nrow(annotations),
    function(ann){
      path_raw <-  annotations$path_raw[ann]
      mp3_dir <-  annotations$mp3_dir[ann]
      
      ffannotations <- annotations$path_annotation[ann] %>% 
        read.table() %>% 
        rename("start" = V1, "end" = V2, "classification" = V3) %>% 
        mutate(
          path_raw = path_raw,
          
          basename = paste0(
            annotations$name_raw[ann] %>% 
              str_remove(fixed(".mp3")),
            "_s",
            floor(start),
            "_",
            classification
          ),
          
          mp3_out = paste0(mp3_dir, "/", basename, ".mp3"),
          wav_out = str_replace_all(mp3_out, "mp3", "wav"),
          
          snip_command = paste0(
            "ffmpeg -y -i ",
            "\"", path_raw, "\"",
            
            " -vn ",
            
            "-ac 1 ",
            
            "-ss ",
            start,
            
            " -to ",
            
            end,
            
            " -c copy ",
            
           "\"", mp3_out, "\""
          ),
          
          wav_command = paste0(
            "ffmpeg -y -i ",
            
            "\"", mp3_out, "\"",
            
            " -ar 16000 ",
            
            "-af highpass=f=200 ",
            
            "-c:a pcm_s16le ",
            
            "\"", wav_out, "\""
          )
        )
    }
  ) %>% 
    bind_rows()
  
#
# Snip to mp3 ----
#
  # make directories
  annotations$mp3_dir %>% 
    unique() %>% 
    sapply(function(dp) dir.create(dp, recursive = T))
  
  mp3_processed <- list.files(
    path = project_root,
    pattern = "mp3 snips",
    include.dirs = T,
    recursive = T,
    full.names = T
  ) %>% 
    sapply(function(dp)list.files(path = dp, full.names = T, recursive = T), simplify = F) %>% 
    unlist()
  
  mp3_subset <- ffcommands %>%
    filter(!(mp3_out %in% mp3_processed)) %>%
    # filter(classification == "bee") %>%
    slice(sample(1:n()))  # shuffles data frame
  

  mp3_subset %>%
    .$snip_command %>%
    mclapply(system, mc.cores = threads)
  
  mp3_processed <- list.files( # re-create mp3_processed to account for newly processed files (needed for knowing what wavs to conver)
    path = project_root,
    pattern = "mp3 snips",
    include.dirs = T,
    recursive = T,
    full.names = T
  ) %>% 
    sapply(function(dp)list.files(path = dp, full.names = T, recursive = T), simplify = F) %>% 
    unlist()
  
#
# Convert ----
#
  # make directories
  annotations$wav_dir %>% 
    unique() %>% 
    sapply(function(dp) dir.create(dp, recursive = T))
  
  wav_processed <- list.files(
    path = project_root,
    pattern = "wav snips",
    include.dirs = T,
    recursive = T,
    full.names = T
  ) %>% 
    sapply(function(dp)list.files(path = dp, full.names = T, recursive = T), simplify = F) %>% 
    unlist()
  
  wav_subset <- ffcommands %>% 
  filter(mp3_out %in% mp3_processed, !(wav_out %in% wav_processed)) %>%
  # filter(classification == "bee") %>%
  slice(sample(1:n())) # shuffles data frame
  

  wav_subset %>%
    .$wav_command %>%
    mclapply(system, mc.cores = threads)
    
  
#
# Save full metadata
#
  annotations$fold <- rep(1:5, times = ceiling(nrow(annotations)/5)) %>% 
    sample() %>% 
    .[1:nrow(annotations)] # this assigns folds completely at random, but with a close-to-equal number of each fold number in total
  
  annotations <- bind_cols(
    annotations,
    annotations$path_raw %>% 
      str_split(pattern = "/", simplify = F) %>% 
      lapply(
        X = .,
        FUN = function(splitpath){data.frame(
          experiment = splitpath[length(splitpath) - 4],
          site = splitpath[length(splitpath) - 2],
          recorder = splitpath[length(splitpath) - 1],
          file = splitpath[length(splitpath) - 0]
        )}
      ) %>% 
      bind_rows()
  )
  
  
  metadata <- ffcommands %>% 
    select(start, end, classification, path_raw, basename) %>% 
    left_join(
      y = annotations %>% 
        select(path_raw, fold, experiment, site, recorder)
    ) %>% 
    mutate(
      duration = end-start,
      filename = paste0(basename, ".mp3")
    ) %>% 
    rename("category" = classification) %>% 
    select(
      experiment, site, recorder, filename, category, fold, duration
    )
  
  write.csv(
    metadata,
    "./training/metadata_HPF200_weighted.csv",
    row.names = F
  )
  