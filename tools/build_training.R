#
# Setup ----
#
  
  library(dplyr)
  library(stringr)
  
  project_root <- "./localData"
  
#
# Get snip info ----
#
    
  dir_snip <- list.dirs(project_root, full.names = T, recursive = T) %>% 
    .[str_detect(string = ., pattern = "wav snips$")]
  
  snipdata <- data.frame(
    source = list.files(dir_snip, full.names = T, recursive = T)
  ) %>% 
    mutate(
      
      experiment = str_extract(
        source,
        pattern = ".*/(.*)/.*/(.*)/(.*)/.*$",
        group = 1
      ),
      
      site = str_extract(
        source,
        pattern = ".*/(.*)/.*/(.*)/(.*)/.*$",
        group = 2
      ),
      
      recorder = str_extract(
        source,
        pattern = ".*/(.*)/.*/(.*)/(.*)/.*$",
        group = 3
      ),
      
      
      filename_raw = str_extract(
        source,
        pattern = ".*/(.*)_s\\d+.*$",
        group = 1
      ) %>% 
        paste0(.,".mp3"),
      
      snipfile = lapply(source, function(fp){str_split(fp, pattern = "/") %>% unlist() %>%  .[[length(.)]]}) %>% 
        unlist(),
      destination = paste0("./training/audio/", snipfile),
      class = str_extract(snipfile, "^.*_(.+)\\.wav", group = 1)
    )
  
#
# Build metadata ----
#
  
  metameta <- snipdata %>% 
    select(experiment, site, recorder, filename_raw) %>% 
    unique()
  
  metameta$fold <- rep(1:5, times = ceiling(nrow(metameta)/5)) %>% 
    sample() %>% 
    .[1:nrow(metameta)] # this assigns folds completely at random, but with a close-to-equal number of each fold number in total
  
  metadata <- snipdata %>% 
    select(experiment, site, recorder, filename_raw, snipfile, class) %>% 
    left_join(metameta) %>% 
    select(!filename_raw) %>% 
    rename("filename" = snipfile, "category" = class)


#
# File operations
#
  
  write.csv(metadata, "./training/metadata_fullwav.csv", row.names = F)
  file.copy(snipdata$source, snipdata$destination)
