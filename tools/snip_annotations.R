library(dplyr)
library(stringr)

project_directory = "./localData"

df <- data.frame(
  annotation = list.files(path = project_directory, recursive = T, full.names = T, pattern = "combinedAnnotations")
) %>% 
  mutate(
    raws = annotation %>% 
      str_replace("combined annotations", "raw audio") %>% 
      str_replace(fixed("combined_annotations.txt"), ".mp3"),
    
    snip_directory = annotation %>% 
      str_replace("combined annotations", "annotated snips") %>% 
      # lazy way to kill the filenames off the end of the path
      str_split("/", simplify = F) %>% 
      lapply(function(str){str[1:(length(str)-1)]}) %>% 
      sapply(function(str) paste0(str, collapse = "/"))
  )

# make directories
df$snip_directory %>% 
  unique() %>% 
  sapply(function(dp) dir.create(dp, recursive = T))

sapply(
  1:nrow(annotations),
  FUN = function(){
    annotation <- read.table(annotations[[obs]])
    names(annotation) <- c("start", "end", "classification")
    
    
    
    sapply(
      1:nrow(annotation),
      function(obs){
        filename
      }
    )
  }
)