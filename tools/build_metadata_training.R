library(dplyr)
library(stringr)

# Import audio data
#
  
  
  audio_path <- "./datasets/bee_training/audio"
  
  files_audio <- list.files(audio_path)
  
  metadata <- data.frame(
    filename = files_audio
  ) %>% 
    mutate(
      fold = 1, # I think? for training?
      category = str_split(filename, pattern = "_", simplify = T)[,1]
    )
  
# Create category numbers for new data
#

  categories <- read.csv("./datasets/ESC-50-master/meta/esc50.csv") %>% 
    .[c("target", "category")] %>% 
    unique() %>% 
    arrange(target)
  
  categories_new <- metadata$category %>% 
    unique() %>% 
    {.[!(.%in%categories$category)]}
  
  categories <- categories %>% 
    rbind(
      data.frame(
        category = categories_new,
        target = -1
      )
    )
  
  categories$target <- 0:(nrow(categories) - 1)

# Build metadata
#
  
  # metadata$src_file <- "" # don't think I need this, looks like it's just tracking for the ESC-50 dataset
  metadata <- metadata %>% 
    mutate(
      target = categories$target[match(x = metadata$category, table = categories$category)],
      .after = fold
    )
  
# Write
#
  
  write.csv(metadata, "./training/metadata_bee.csv", row.names = F)
  