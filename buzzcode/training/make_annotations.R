library(dplyr)
library(stringr)
annotations <- read.csv('./training/annotations/annotations_all.csv')

# Clean (drop useless labels)
#
  clean <- annotations %>% 
    filter(classification!='RECLASSIFY')
  write.csv(clean, './training/annotations/annotations_clean.csv', row.names = F)
  
# Drop traffic (remove ambiguous and numerous "traffic" labels that could be any mech_auto or mech_plane)
#
  droptraffic <- clean %>% 
    filter(classification!='mech_traffic_RECLASSIFY')
  
  write.csv(clean, './training/annotations/annotations_clean.csv', row.names = F)

# Write a blank conversion csv for creating new conversions
#
  write.csv(
    data.frame(
      from = annotations$classification %>% unique() %>% sort(),
      to = ''
    ),
    
    'conversion_blank.csv'
  )
