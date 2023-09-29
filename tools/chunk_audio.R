library(dplyr)
library(stringr)

dir_in = "./localData/audio_superhour"
dir_out = "./localData/audio_superhour_chunk"

chunker <- function(dir_in, dir_out, segment_length = (60*60)){
  paths_in <- list.files(
    path = dir_in,
    pattern = ".mp3$",
    recursive = T,
    full.names = T
  )
  
  names_in <- paths_in %>% 
    str_split(pattern = "/", simplify = F) %>% 
    lapply(X = ., FUN = function(path){path[length(path)]}) %>% 
    unlist()
  
  filedf <- data.frame(
    paths_in = paths_in
  ) %>% 
    mutate(
      paths_out = str_replace(string = paths_in, pattern = fixed(dir_in), replacement = dir_out) %>% 
        str_replace(".mp3", "-%03d.mp3") # this tags the command with a string that ffmpeg needs to name files incrementally
    )
  
  newdirs <- str_remove(string = paths_in, pattern = names_in) %>% 
    str_replace(pattern = dir_in, replacement = dir_out) %>% 
    unique()
  
  sapply(newdirs, function(d) dir.create(d, recursive = T))
  
  mapply(
    file_in = filedf$paths_in,
    file_out= filedf$paths_out,
    FUN = function(file_in, file_out){
      command <- paste0("ffmpeg -i ", file_in, " -f segment -segment_time ", segment_length, " -c copy ", file_out)
      system(command)
    }
  )
}