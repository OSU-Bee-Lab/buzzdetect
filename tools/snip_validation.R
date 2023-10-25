#
# Setup ----
#

  library(dplyr)
  library(stringr)
  library(parallel)
  
  validation_dir = "./localValidation"
  threads <- 7
  
  modelname <- "HPF200_weighted"
  
  snip_dir <- "long bee snips"
  buffer <- 4 # in seconds, applied in both directions (added length = 2*buffer)

#
# Metadata ----
#
  metadata <- read.csv(
    file = paste0("./models/", modelname, "/metadata.csv")
  ) %>% 
    mutate(
      path_raw = paste(
        "./localValidation",
        experiment,
        "raw audio",
        site,
        recorder,
        filename %>% 
          str_replace(
            pattern = "_s.*",
            replacement = ""
          ) %>% 
          paste0(., ".mp3"),
        sep = "/"
      )
    )
  
  
#
# Master ----
#
  
  annotations <- data.frame(
    path_annotation = list.files(path = validation_dir, recursive = T, full.names = T, pattern = "combinedAnnotations")
  ) %>% 
    mutate(
      path_raw = path_annotation %>% 
        str_replace("combined annotations", "raw audio") %>% 
        str_replace(fixed("_combinedAnnotations.txt"), ".mp3"),
      
      name_raw = path_raw  %>% 
        str_split("/", simplify = T) %>% 
        .[,ncol(.)],
      
      mp3_dir = path_annotation %>% 
        str_replace("combined annotations", snip_dir) %>% 
        # lazy way to kill the basenames off the end of the path
        str_split("/", simplify = F) %>% 
        lapply(function(str){str[1:(length(str)-1)]}) %>% 
        sapply(function(str) paste0(str, collapse = "/"))
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
          start = start - buffer,
          end = end + buffer,
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
          )
        )
    }
  ) %>% 
    bind_rows() %>% 
    left_join(
      metadata %>% 
        select(path_raw, fold) %>% 
        unique(),
      by = "path_raw"
    )
  
#
# Snip to mp3 ----
#
  # make directories
  annotations$mp3_dir %>% 
    unique() %>% 
    sapply(function(dp) dir.create(dp, recursive = T))
  
  mp3_processed <- list.files(
    path = validation_dir,
    pattern = snip_dir,
    include.dirs = T,
    recursive = T,
    full.names = T
  ) %>% 
    sapply(function(dp)list.files(path = dp, full.names = T, recursive = T), simplify = F) %>% 
    unlist()
  
  mp3_subset <- ffcommands %>%
    filter(!(mp3_out %in% mp3_processed)) %>%
    filter(classification == "bee", fold == 5) %>%
    slice(sample(1:n()))   # shuffles data frame
  

#
# RUN ----
#
  mp3_subset %>%
    .$snip_command %>%
    mclapply(system, mc.cores = threads)
  