library(dplyr)
library(stringr)

#
# Setup ----
#
  project_root <- "./localData/manual annotations/Bee Audio 2022 Original"

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
  
  names(annotation_files)[c(1, ncol(annotation_files) - c(2, 1, 0))] <- c("filepath", "site", "recorder", "filename")
  
  annotation_data <- sapply(
    simplify = F,
    1:nrow(annotation_files),
    FUN = function(f){
      read.csv(file = annotation_files$filepath[f], sep = "\t", header = F) %>% 
        mutate(
          site = annotation_files$site[f],
          recorder = annotation_files$recorder[f],
          chunk = annotation_files$filename[f]
        )
    }
  ) %>% 
    bind_rows()
  
  names(annotation_data) <- c("start", "end", "classification", "site", "recorder", "chunk")
  
  annotation_data$time_chunkStart <- str_extract(
    annotation_data$chunk,
    "^.*_(\\d+)_(\\d+)_manual.*$", # pull the first number and second number
    group = c(1, 2)
  ) %>% 
    {paste(.[1], .[2], sep = " ")} %>% # combine the first and second number into one string, with a space separating
    as.POSIXct(
      format = "%Y%m%d %H%M%S", # format as a date
      tz = "EST"
    )
  
  annotation_data$start_realtime_approximate <- annotation_data$time_chunkStart + annotation_data$start
  annotation_data$end_realtime_approximate <- annotation_data$time_chunkStart + annotation_data$end

#
# Pull raw file info ----
#

  raw_files <- data.frame(
    filepath = list.files(
      project_root,
      pattern = "\\.mp3$",
      recursive = T,
      full.names = T
    )
  ) %>% 
    {bind_cols(
      .,
      str_split(.$filepath, pattern = "/", simplify = T) %>% 
        data.frame()
    )}
  
  names(raw_files)[c(1, ncol(raw_files) - c(2, 1, 0))] <- c("filepath", "site", "recorder", "filename")
  
  raw_files$time <- str_extract(
    raw_files$filename,
    "^(\\d+)_(\\d+).*", 
    group = c(1, 2)
  ) %>% 
    {paste(.[,1], .[,2], sep = " ")} %>% 
    as.POSIXct(
      format = "%y%m%d %H%M", 
      tz = "EST"
    )


#
# combine! ----
#
  
  
  combine <- function(site_in, recorder_in){
    # work within a set; all annotations greater than the last time WITHIN the set belongs to that recording file
    set <- raw_files %>%
      filter(site == site_in, recorder == recorder_in) %>% 
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
        
        annotations <- annotation_data %>% 
          filter(
            site == obs$site,
            recorder == obs$recorder,
            start_realtime_approximate > obs$time,
            end_realtime_approximate < (obs$time + duration_total)
          )  %>% 
          mutate(
            difftime_chunk_raw = difftime(time_chunkStart, obs$time, units = "secs") %>% 
              as.numeric(),
            
            start_adjusted = difftime_chunk_raw + start,
            end_adjusted = difftime_chunk_raw + end,
            raw_file = obs$filepath,
            distance_from_end = duration_total - end_adjusted
          )
        
        return(annotations)
      }
    )
    
    names(setData) <- set$filepath
    
    return(setData)
  }
  
  siteRecorders <- raw_files %>% 
    select(site, recorder) %>% 
    unique()

  data <- mapply(
    SIMPLIFY = F,
    site_in = siteRecorders$site,
    recorder_in = siteRecorders$recorder,
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
          select(start_adjusted, end_adjusted,  classification) %>% 
          rename("start" = start_adjusted, "end" = end_adjusted)
      }
    )
  
  # write annotations
  sapply(
    raw_files$filepath,
    function(fp){
      path_annotation <- fp %>% 
        str_replace(fixed(".mp3"), "_combinedAnnotations.txt")
      
      if(nrow(annotations[[fp]]) == 0){return(NULL)}
      
      names(annotations[[fp]]) <- NULL
      write.table(x = annotations[[fp]], file = path_annotation, sep = "  ", row.names = F, col.names = F, quote = F)
    }
  )
  
#   
# #
# # Graveyard - functions to check for errors ----
# #
# 

# 
#   annotation_data$used <- mapply(
#     SIMPLIFY = F,
#     start_in = annotation_data$start,
#     end_in = annotation_data$end,
#     site_in = annotation_data$site,
#     recorder_in = annotation_data$recorder,
#     FUN = function(start_in, end_in, site_in, recorder_in){
#       sub <- filter(data_collapsed, start == start_in, end == end_in, site == site_in, recorder == recorder_in)
#       return(nrow(sub))
#     }
#   ) %>%
#     unlist()
# 
