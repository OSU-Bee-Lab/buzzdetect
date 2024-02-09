library(dplyr)
library(stringr)


metadata_list <- list.files("./training/metadata", pattern = ".csv", full.names = T) %>% 
  sapply(read.csv, simplify = F)

quicksumm <- function(metadata){
  metadata %>% 
    filter(fold != 5) %>% 
    group_by(
      classification
    ) %>% 
    summarize(frames_total = sum(frames))
}


weight_functions <- list(
  invProp = function(metadata){
    metadata %>% 
      quicksumm() %>% 
      mutate(
        weight = (sum(frames_total)/frames_total) %>% 
          log()
      )
  }
)

execution_grid <- expand.grid(
  metadata = names(metadata_list),
  func = names(weight_functions),
  stringsAsFactors = F
)

weights <- mapply(
  metadata = execution_grid$metadata,
  func = execution_grid$func,
  function(metadata, func){
    weight = weight_functions[[func]](metadata_list[[metadata]])
    meta_name <- basename(metadata) %>% 
      str_extract("metadata_(.*)\\.csv$", group = 1)
    
    weightpath <- paste0("./training/weights/", func, "Weight_", meta_name, "Metadata.csv")

    write.csv(weight, weightpath)
    
    return(weight)
  },
  SIMPLIFY = F
)

