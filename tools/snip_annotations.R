#
# Setup ----
#

  library(dplyr)
  library(stringr)
  library(parallel)
  
  project_root = "./localData"
  
#
# Master ----
#
  
  df <- data.frame(
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
        # lazy way to kill the filenames off the end of the path
        str_split("/", simplify = F) %>% 
        lapply(function(str){str[1:(length(str)-1)]}) %>% 
        sapply(function(str) paste0(str, collapse = "/")),
      
      wav_dir = mp3_dir %>% 
        str_replace("mp3", "wav")
    )
  
  ffcommands <- sapply(
    simplify = F,
    1:nrow(df),
    function(ann){
      path_raw <-  df$path_raw[ann]
      mp3_dir <-  df$mp3_dir[ann]
      
      ffdf <- df$path_annotation[ann] %>% 
        read.table() %>% 
        rename("start" = V1, "end" = V2, "classification" = V3) %>% 
        mutate(
          filename = paste0(
            df$name_raw[ann] %>% 
              str_remove(fixed(".mp3")),
            "_s",
            floor(start),
            "_",
            classification,
            ".mp3"
          ),
          
          mp3_out = paste0(mp3_dir, "/", filename),
          wav_out = str_replace_all(mp3_out, "mp3", "wav"),
          
          snip_command = paste0(
            "ffmpeg -y -i ",
            "\"", path_raw, "\"",
            
            " -vn ",
            
            "-ar 16000 ",
            
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
            
            " -c:a pcm_s16le ",
            
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
  df$mp3_dir %>% 
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
    slice(sample(1:n())) %>%  # shuffles data frame
    .[1:100,]
  
  mp3_subset %>%
    .$snip_command %>%
    mclapply(system, mc.cores = 7)
  
#
# Convert ----
#
  # make directories
  df$wav_dir %>% 
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
  filter(!(wav_out %in% wav_processed)) %>%
  # filter(classification == "bee") %>%
  slice(sample(1:n())) # shuffles data frame
  
  wav_subset %>%
    .$wav_command %>%
    mclapply(system, mc.cores = 7)
    