library(dplyr)
library(stringr)

n_folds <- 5

annotations <- read.csv('./training/annotations/annotations_all.csv')

ident <- annotations  %>% 
  .$ident %>% 
  unique() %>% 
  sort()

folds <- data.frame(ident)
info <- str_split(ident, pattern = '/', simplify = T) %>% 
  as.data.frame() %>% 
  rename(
    'experiment'= V2,
    'site' = V3,
    'recorder' = V4,
    'timestamp' = V5
  ) %>% 
  select(!V1)

folds <- bind_cols(folds, info)


df <- lapply(
  X = unique(folds$experiment),
  FUN = function(experiment_in){
    folds %>% 
      select(!c(timestamp, ident)) %>% 
      filter(experiment == experiment_in) %>% 
      unique() %>% 
      mutate(
        fold = rep(1:n_folds, ceiling(n()/n_folds))[1:n()]
      )
  }
) %>% 
  bind_rows()
  
  
folds <- folds %>% 
  left_join(df)

write.csv(folds, './training/folds.csv', row.names = F)
