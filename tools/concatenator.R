library(dplyr)
library(stringr)

raw_dir <- "./output"

df <- data.frame(
  path = list.files(raw_dir, pattern = fixed("_results.txt"), recursive = T, full.names = T)
) %>% 
  bind_cols(
    lapply(
      .$path,
      function(fp){
        path_split <- str_split(fp, pattern = "/", simplify = T)
        annotation_parent <- fp %>%
          str_replace("_s\\d+_results.txt", replacement = "_processedAnnotations.txt")
          
        
        return(data.frame(recorder, dir, mp3_filename))
      }
    ) %>% 
      bind_rows
  )

melt_annotations <- function(mp3_in){
  
}
  
str_extract(raw_df$filename, pattern = "(\\d+)_(\\d+)_s(\\d+)_results\\.txt", group = c(1, 2, 3)) %>% 
  data.frame() %>% 
  rename()

concatenator <- function(data_in, confidence_threshold = 1, isolation_tolerance = 1){
  frameHop <- (data_in$start - lag(data_in$start)) %>% 
    .[2:length(.)] %>% 
    unique()
  
  if(length(frameHop) != 1){stop("Multiple frame hop sizes detected; cannot proceed")}
  
  isolation_frames <- isolation_tolerance/frameHop
  
  is_isolated <- function(observationNum){
    data_in$classification[]
  }
}