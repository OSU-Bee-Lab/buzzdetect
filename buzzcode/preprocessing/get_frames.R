get_frames <- function(duration_in){
  if(duration_in < 0.960){
    warning("sub-frame duration, returning 0")
    return(0)
  }
  
  # primary frame stream
  n_frames_prime <- seq(0, duration_in, 0.960) %>% 
    .[-1] %>% 
    length()
  
  # alternate frame stream
  n_frames_sec <- seq(0.480, duration_in, 0.960) %>% 
    .[-1] %>% 
    length()
  
  return(sum(n_frames_prime, n_frames_sec))
}
