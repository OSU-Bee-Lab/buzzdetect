#
# Setup ----
#

  library(dplyr)
  library(stringr)
  library(parallel)
  
  project_root = "./localData"
  
#
# ----
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
      
      snip_dir = path_annotation %>% 
        str_replace("combined annotations", "annotated snips") %>% 
        # lazy way to kill the filenames off the end of the path
        str_split("/", simplify = F) %>% 
        lapply(function(str){str[1:(length(str)-1)]}) %>% 
        sapply(function(str) paste0(str, collapse = "/"))
    )
  
  ffcommands <- sapply(
    simplify = F,
    1:nrow(df),
    function(ann){
      path_raw <-  df$path_raw[ann]
      snip_dir <-  df$snip_dir[ann]
      
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
            ".wav"
          ),
          
          path_out = paste0(snip_dir, "/", filename),
          
          command = paste0(
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
           "\"", path_out, "\""
          )
        )
    }
  ) %>% 
    bind_rows()

#
# Snip audio files ----
#
  
  # make directories
  df$snip_dir %>% 
    unique() %>% 
    sapply(function(dp) dir.create(dp, recursive = T))
  
  
  processed <- list.files(
    path = project_root,
    pattern = "annotated snips",
    include.dirs = T,
    recursive = T,
    full.names = T
  ) %>% 
    sapply(function(dp)list.files(path = dp, full.names = T, recursive = T), simplify = F) %>% 
    unlist()
  
  subset <- ffcommands %>% 
    filter(!(path_out %in% processed)) %>%
    filter(classification == "bee") %>%
    slice(sample(1:n())) # shuffles data frame
  
  subset %>%
    .$command %>% 
    mclapply(system, mc.cores = 7)
    