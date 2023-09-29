library(dplyr)

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