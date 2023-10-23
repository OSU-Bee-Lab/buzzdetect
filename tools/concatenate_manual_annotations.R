library(dplyr)
library(stringr)

#
# Setup ----
#
  project_root <- "./localData"
  
  regex_chunk <- c(
      "Bee Audio 2022 Original" = "^.*_(\\d+)_(\\d+)_manual.*$",
      "Karlan Forrester - Soybean Attractiveness" = "^chunk(\\d+)_(\\d+).*$"
  )
  
#
# Conditional modifications
#
  
  alignmentDF <- read.csv(paste0(project_root, "/correction.csv"))
  
  # hmm...these buffers should really be applied during the snipping; this script is meant only to combine annotations, not change them
  default_buffer <- 0.50
  
  buffers <- c(
    "bee" = 1.75
  )

#
# Pull annotation data ----
#
  
  annotation_files <- data.frame(
    filepath = list.files(
      project_root,
      pattern = fixed("manual"),
      recursive = T,
      full.names = T
    )
  ) %>% 
    {bind_cols(
      .,
      str_split(.$filepath, pattern = "/", simplify = T) %>% 
        data.frame()
    )}
  
  names(annotation_files)[c(1, 4, ncol(annotation_files) - c(2, 1, 0))] <- c("filepath", "experiment", "site", "recorder", "filename")
  
  annotation_data <- sapply(
    simplify = F,
    1:nrow(annotation_files),
    FUN = function(f){
      read.csv(file = annotation_files$filepath[f], sep = "\t", header = F) %>% 
        mutate(
          experiment = annotation_files$experiment[f],
          site = annotation_files$site[f],
          recorder = annotation_files$recorder[f],
          chunk = annotation_files$filename[f]
        )
    }
  ) %>% 
    bind_rows() %>% 
    rename("start" = V1, "end" = V2, "classification" = V3) %>% 
    mutate(
      time_chunkStart = str_extract(
        chunk,
        regex_chunk[experiment],
        group = c(1, 2)
      ) %>% 
        {paste(.[,1], .[,2], sep = " ")} %>% # combine the first and second number into one string, with a space separating
        as.POSIXct(
          format = "%Y%m%d %H%M%S", # format as a date
          tz = "EST"
        ),
      
      start_realtime_approximate = time_chunkStart + start,
      end_realtime_approximate = time_chunkStart + end
    )

#
# Pull raw file info ----
#
  raw_dirs <- list.dirs(project_root, recursive = F) %>% 
    paste0(., "/raw audio")
  
  raw_files <- data.frame(
    filepath = list.files(
      raw_dirs,
      pattern = "\\.mp3$",
      recursive = T,
      full.names = T
    )
  ) %>% 
    {bind_cols(
      .,
      str_split(.$filepath, pattern = "/", simplify = T) %>% 
        data.frame()
    )} %>% 
    rename(
      "experiment" = X3, "site" = X5, "recorder" = X6, "filename" = X7
    ) %>% 
    mutate(
      time = str_extract(
        filename,
        "^(\\d+)_(\\d+).*", 
        group = c(1, 2)
      ) %>% 
        {paste(.[,1], .[,2], sep = " ")} %>% 
        as.POSIXct(
          format = "%y%m%d %H%M", 
          tz = "EST"
        )
    )
  
#
# combine! ----
#
  
  
  combine <- function(experiment_in, site_in, recorder_in){
    # work within a set; all annotations greater than the last time WITHIN the set belongs to that recording file
    set <- raw_files %>%
      filter(experiment == experiment_in, site == site_in, recorder == recorder_in) %>% 
      arrange(time)
    
    setData <- sapply(
      simplify = F,
      X = 1:nrow(set),
      FUN = function(observation_in){
        obs <- set[observation_in,]
        
        duration_raw <- system2(
          command = "ffprobe",
          args = paste0("\"", obs$filepath, "\""),
          stdout = T,
          stderr = T
        ) %>% 
          {.[str_detect(string = ., pattern ="Duration: ")]} %>% 
          str_extract(pattern = ".*Duration: (\\d*):(\\d*):(\\d*\\.\\d*).+", group = 1:3) %>% 
          as.numeric()
        
        duration_total <- (((duration_raw[1] * 60) + duration_raw[2]) * 60) + duration_raw[3]
        
        alignment_correction <- alignmentDF %>% 
          filter(experiment == experiment_in, site == site_in, recorder == recorder_in, raw_file == obs$filename) %>%
          .$adjustment_to_labels %>% 
          ifelse(length(.) == 0, 0, .)
          
        
        annotations <- annotation_data %>% 
          filter(
            site == obs$site,
            recorder == obs$recorder,
            start_realtime_approximate > obs$time,
            end_realtime_approximate < (obs$time + duration_total)
          )  %>% 
          mutate(
            difftime_chunk_raw = difftime(time_chunkStart, obs$time, units = "secs") %>% 
              as.numeric() %>%
              {. + alignment_correction},
            
            class_buffer = buffers[classification] %>% 
              ifelse(is.na(.), default_buffer, .),
            
            start_adjusted = ((difftime_chunk_raw + start) - class_buffer) %>% 
              ifelse(. < 0, 0, .), # expand the annotation by 1s on either side;
            end_adjusted = (difftime_chunk_raw + end) + class_buffer, # the south charleston annotations do not accurately capture the entire bee sound
            raw_file = obs$filepath,
            distance_from_end = duration_total - end_adjusted
          )
        
        return(annotations)
      }
    )
    
    names(setData) <- set$filepath
    
    return(setData)
  }
  
  uniqueRecorders <- raw_files %>% 
    select(experiment, site, recorder) %>% 
    unique()

  data <- mapply(
    SIMPLIFY = F,
    experiment_in = uniqueRecorders$experiment,
    site_in = uniqueRecorders$site,
    recorder_in = uniqueRecorders$recorder,
    FUN = combine
  )  
  
  data_collapsed <- data %>%
    unlist(recursive = F) %>%
    bind_rows()
  
  annotations <- raw_files$filepath %>% 
    sapply(
      simplify = F,
      FUN = function(fp){
        data_collapsed %>% 
          filter(raw_file == fp) %>% 
          arrange(start_adjusted) %>% 
          select(start_adjusted, end_adjusted,  classification)
      }
    )

#  
# write ----
# 
  # make directories
  list.dirs(path = raw_dirs, recursive = T) %>% 
    str_replace("raw audio", "combined annotations") %>% 
    sapply(function(dp) dir.create(dp, recursive = T))
  
  # write files
  sapply(
    raw_files$filepath,
    function(fp){
      path_annotation <- fp %>% 
        str_replace(fixed(".mp3"), "_combinedAnnotations.txt") %>% 
        str_replace("raw audio", "combined annotations")
      
      if(nrow(annotations[[fp]]) == 0){return(NULL)}
      
      names(annotations[[fp]]) <- NULL
      write.table(x = annotations[[fp]], file = path_annotation, sep = "\t", row.names = F, col.names = F, quote = F)
    }
  )
  