library(dplyr)
library(stringr)
source('./buzzcode/preprocessing/get_frames.R')
  
  # TO DO: drop counting frames, to make buzzdetect embedder-flexible
  

  path_raw <- './training/metadata/metadata_raw.csv'
  metadata_raw <- read.csv(path_raw)
  drop_threshold <- 15  # drop below 15 obs; need at least 3 test files
  # In theory I should use frames, not files, but I don't think it'll make a difference with small files
  
  quicksumm <- function(metadata){
    metadata %>% 
      group_by(classification) %>% 
      summarize(count = n(), duration = sum(duration), frames = sum(frames))
  }
  
  quicksub <- function(metadata){
    summary <- quicksumm(metadata)
    classes <- filter(summary, count >= drop_threshold)$classification
    out <- filter(metadata, classification %in% classes) %>% 
      mutate(start_floor = floor(start)) %>% 
      group_by(ident, start_floor) %>% 
      mutate(count=n())
    
    dupes <- filter(out, count > 1)
    if(nrow(dupes) > 1){
      # to do: return which labels are duped
      warning('multiple labels detected for same audio regions; dropping duplicates')
    }
    
    out <- out %>% 
      filter(count == 1) %>% 
      ungroup()
    
    return(select(out, !c(count, start_floor)))
  }
  
  assign_folds <- function(metadata){
    metadata_perclass <- lapply(
      unique(metadata$classification),
      function(c){
        meta_sub <- filter(metadata, classification == c)
        meta_sub$fold <- runif(n = nrow(meta_sub), min = 1, max = 6) %>%  
          floor()
        
        return(meta_sub)
      }
    )
    
    metadata_out <- bind_rows(metadata_perclass)
    
    return(metadata_out)
  }
  
  
# Give raw metadata folds, if it doesn't already have it
#
  if(!('fold' %in% names(metadata_raw))){
    metadata_raw <- assign_folds(metadata_raw)
    write.csv(metadata_raw, path_raw, row.names = F)
  }
  
  if(!('frames' %in% names(metadata_raw))){
    metadata_raw$frames <- sapply(metadata_raw$duration, get_frames)
    write.csv(metadata_raw, path_raw, row.names = F)
  }
  
# Summarize raw metadata for future use
#
  summ_raw <- quicksumm(metadata_raw)
  
#
# Making child metadata ----
#
  
  # an attempt at balancing available data and useful classification
  #
  metadata_intermediate <- metadata_raw %>% 
    filter(!(classification %in% c("mech", "RECLASSIFY", "ins_buzz_RECLASSIFY", "mech_auto_RECLASSIFY"))) %>% 
    mutate(
      classification_original = classification,
      classification = case_when(
        classification_original == "ambient_bang" ~ "ambient_scraping",  # merge scraping and bang; they're somewhat similar\
        classification_original == "ambient_rustle" ~ "ambient_scraping", 
        str_detect(classification_original, "mech_auto") ~ "mech_auto", # combine automobile sounds
        str_detect(classification_original, "mech_plane") ~ "mech_plane",
        T ~ classification_original
      )
    ) %>% 
    quicksub() %>% 
    assign_folds()
  
  summ_intermediate <- quicksumm(metadata_intermediate)
  
  write.csv(metadata_intermediate, "./training/metadata/metadata_intermediate.csv", row.names = F)
  
  
  # a metadata where ambient day and night are combined (attempt to combat weird phenomenon of model attraction to ambient_night)
  #
  metadata_strictish <- metadata_raw %>% 
    filter(!str_detect(classification, "RECLASSIFY")) %>% 
    mutate(
      classification_original = classification,
      classification = case_when(
        classification_original %in% c('ambient_day', 'ambient_night') ~ 'ambient_sound',
        T ~ classification_original
      )
    ) %>% 
    quicksub() %>% 
    assign_folds()
  
  write.csv(metadata_strictish, "./training/metadata/metadata_commonambient.csv", row.names = F)
  
    
  
  # a strict metadata using only finalized classifications
  #
    metadata_strict <- metadata_raw %>% 
      filter(!str_detect(classification, "RECLASSIFY")) %>% 
      quicksub() %>% 
      assign_folds()
    
    summ_strict <- quicksumm(metadata_strict)
    
    write.csv(metadata_strict, "./training/metadata/metadata_strict.csv", row.names = F)
    
      