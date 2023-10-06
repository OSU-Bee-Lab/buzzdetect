library(dplyr)
library(stringr)

project_directory = "./localData"

df <- data.frame(
  path_annotation = list.files(path = project_directory, recursive = T, full.names = T, pattern = "combinedAnnotations")
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

# make directories
df$snip_dir %>% 
  unique() %>% 
  sapply(function(dp) dir.create(dp, recursive = T))

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
          classification,
          "_",
          df$name_raw[ann] %>% 
            str_remove(fixed(".mp3")),
          "_",
          floor(start),
          ".mp3"
        ),
        
        path_out = paste0(snip_dir, "/", filename),
        
        command = paste0(
          "ffmpeg -i ",
          "\"", path_raw, "\"",
          
          " -ss ",
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

ffcommands %>% 
  filter(classification == "bee") %>% 
  .$command %>% 
  sapply(system)
  
