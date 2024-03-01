library(dplyr)
library(parallel)
library(stringr)
library(tidyr)
library(ggplot2)
library(caret)
library(shadowtext)

cpus <- 8
dir_training = './training/audio'

dir_out <- './localData/model_inspection'
dir.create(dir_out, showWarnings = F, recursive = F)

buzz_classes <- c('ins_buzz_bee', 'ins_buzz_high', 'ins_buzz_low')

inspect_model <- function(modelname, write = T){
  print(paste0('LAUNCHING inspection of model ', modelname))
  
  #  filesystem prep
  #
    dir_model <- file.path('models', modelname)
    dir_test <- file.path(dir_model, 'output_testFold')
    
  # loading data
  #
    print('loading data')
    metadata <- read.csv(file.path(dir_model, 'metadata.csv')) %>% 
      mutate(
        stub_audio = path_audio %>% 
          str_remove(dir_training) %>% 
          str_remove(paste0('.',tools::file_ext(.)))
      )
    
    import_data <- function(path_test){
      stub_audio = path_test %>% 
        str_remove(dir_test) %>% 
        str_remove('_buzzdetect\\.csv')
      
      snip_data <- read.csv(path_test) %>% 
        mutate(
          stub_audio = stub_audio
        )
      
      return(snip_data)
    }
    
    paths_test <- list.files(dir_test, recursive = T, full.names = T, pattern = "buzzdetect.csv")
    data <- paths_test %>% 
      lapply(
        FUN = import_data
      ) %>% 
      bind_rows() %>% 
      select(!c(X, start, end, score_predicted))
    
    data <- data %>%
      left_join(
        select(metadata, stub_audio, classification)
      )

    # data_long <- data %>%
    #   pivot_longer(
    #     cols = starts_with("score_"),
    #     names_to = "target_name",
    #     values_to = "target_activation"
    #   ) %>%
    #   mutate(
    #     target_name = str_remove(target_name, "score_"),
    #     is_correct = target_name == classification
    #   ) %>%
    #   filter(!is.na(is_correct))
    # 
    # 
    # data_falseneg <- data %>%
    #   filter(classification == "ins_buzz_bee", class_predicted != "ins_buzz_bee") %>%
    #   mutate(
    #     across(
    #       starts_with("score_"),
    #       ~ .x - score_ins_buzz_bee
    #     )
    #   ) %>%
    #   select(!c(stub_audio)) %>%
    #   pivot_longer(
    #     cols = starts_with("score_"),
    #     names_to = "target_name",
    #     values_to = "target_activation"
    #   )
    
    print('done loading data')

    
  # Confusion
  #
    print('starting confusion')
    levels(metadata$classification) <- metadata$classification %>% unique() %>% sort()

    confusion <- caret::confusionMatrix(
      data = factor(data$class_predicted, levels = levels(metadata$classification)),
      reference = factor(data$classification, levels = levels(metadata$classification))
    )
    
    confusion_table <- confusion$table %>% 
      as.data.frame() %>% 
      rename('actual' = Reference, 'predicted' = 'Prediction', 'frequency' = Freq) %>% 
      group_by(actual) %>% 
      reframe(
        predicted = predicted,
        count = frequency,
        proportion = frequency/sum(frequency)
      ) %>% 
      mutate(
        correct = predicted==actual
      )
    
    levels(confusion_table$actual) <- sort(levels(confusion_table$actual))
    levels(confusion_table$predicted) <- levels(confusion_table$actual)
    
    print('plotting confusion')
    plot_confusion <- ggplot(
      confusion_table %>% 
        mutate(is_zero = proportion < 0.01),
      aes(
        x = actual,
        y = predicted,
        fill = proportion,
        label = round(proportion, 2) %>% 
          format(nsmall=2)
      )
    ) +
      geom_tile(color = "#0d0429", linewidth = 1.3) +
      geom_text(aes(color = is_zero)) +
      scale_x_discrete(position = 'top') +
      scale_y_discrete(limits=rev) +
      scale_color_manual(values = c('TRUE' = '#0d0429', 'FALSE' = 'white'), guide = 'none') +
      scale_fill_gradientn(colors=c('#0d0429', '#cf325f', '#f8ae5e'), guide = 'none') +
      theme(
        panel.background = element_rect(fill="#0d0429"),
        plot.background = element_rect(fill="#0d0429"),
        text = element_text(color = 'white'),
        axis.text.x = element_text(angle = -45),
        axis.text = element_text(color = 'white') 
      ) +
      ggtitle(paste0('confusion table for ', modelname)) +
      geom_tile(
        data = expand.grid(foo = c('ins_buzz_bee', 'ins_buzz_high', 'ins_buzz_low'), bar = c('ins_buzz_bee', 'ins_buzz_high', 'ins_buzz_low'), proportion = 1),
        mapping = aes(x = bar, y = foo),
        fill = "#00000000",
        linewidth = 1.3,
        color = "#212121"
      ) +
      # true buzz highlight
      geom_tile(
        aes(x = 'ins_buzz_bee', y = 'ins_buzz_bee'),
        fill = "#00000000",
        linewidth = 1.3,
        color = "darkgray"
      )
    
    if(write){
      print('saving confusion')
      write.csv(confusion_table, file.path(dir_model, paste0('confusion_', modelname, '.csv')))
      
      ggsave(
        filename = file.path(dir_model, paste0('PLOT_confusion_MODEL_', modelname,'.svg')),
        plot = plot_confusion,
        width = 12,
        height = 9
      )
    }
    
    
    # Sensitivity, specificity ----
    #
      
      get_sensitivity <- function(class_in){
        # sensitivity = p(positive predicted | positive actual)
        sub <- confusion_table %>% 
          filter(actual %in% class_in) %>%  # I'm using %in% so you can check multi-class sensitivity
          filter(actual == predicted)
        
        sensitivity <- mean(sub$proportion)
        
        return(sensitivity)
      }
      
      get_specificity <- function(class_in){
        # specificity = p(negative predicted | negative actual)
        # however, my classes are imbalanced, so great performance in a huge-volume-class will throw off metric
        # I'll use proportions instead; this makes an imperfect real-world specificity; it would be better
        # to weight each class according to it's real-world occurrence, but that's not really possible.
        
        # sub for negative actuals
        sub <- confusion_table %>% 
          filter(!(actual %in% class_in))
        
        # sum the total possible proportions
        total <- sum(sub$proportion)  # total = n(total classes) - n(predicted classes)
        
        # sum the negative prediction proportions
        negatives <- sub %>% 
          filter(!(predicted %in% class_in)) %>% 
          {sum(.$proportion)}
        
        specificity <- negatives/total
        
        return(specificity)
      }
      
      get_specificity_common <- function(class_in){
        # here I want to make a more critical measure of specificity; take the
        # confusion of only the most prevalent classes. The model does great on rain and geese,
        # so what? They're rare. Let's look at the really common classes
        
        # note: won't work for all metadata
        
        # sub for negative actuals of relevant categories
        sub <- confusion_table %>% 
          filter(
            !(actual %in% class_in),  # get negatives
            actual %in% c('ambient_day', 'ambient_scraping') |  # these ambient sounds are the most common I've heard
              str_detect(actual, 'mech_auto') |  # cars are very common
              str_detect(actual, 'mech_plane')  # planes are very common
          )
        
        # sum the total possible proportions
        total <- sum(sub$proportion)  # total = n(common classes)
        
        # sum the negative prediction proportions
        negatives <- sub %>% 
          filter(!(predicted %in% class_in)) %>% 
          {sum(.$proportion)}
        
        specificity_common <- negatives/total
        
        return(specificity_common)
      }
      
      metricsTable <- data.frame(
        class = buzz_classes
      ) %>% 
        rowwise() %>% 
        mutate(
          sensitivity = get_sensitivity(class),
          specificity = get_specificity(class),
          specificity_common = get_specificity_common(class)
        ) %>% 
        bind_rows(
          data.frame(
            class = 'all_buzzes',
            sensitivity = get_sensitivity(buzz_classes),
            specificity = get_specificity(buzz_classes),
            specificity_common = get_specificity_common(buzz_classes)
          )
        )
      
      plot_metrics <- ggplot(
        metricsTable %>% 
          pivot_longer(
            cols = !class,
            names_to = 'metric',
            values_to = 'value'
          ),
        aes(
          x = metric,
          y = class,
          fill = value,
          label = round(value, 2) %>% 
            format(nsmall=2)
        )
      ) +
        geom_tile(color = "#0d0429", linewidth = 1.3) +
        geom_text(color = 'white', size = 4) +
        scale_x_discrete(position = 'top') +
        scale_y_discrete(limits=rev) +
        scale_fill_gradientn(colors=c('#0d0429', '#cf325f', '#f8ae5e'), guide = 'none') +
        theme(
          panel.background = element_rect(fill="#0d0429"),
          plot.background = element_rect(fill="#0d0429"),
          text = element_text(color = 'white'),
          axis.text.x = element_text(angle = -45),
          axis.text = element_text(color = 'white') 
        ) +
        ggtitle(paste0('sensitiviy and specificity for ', modelname))
    
    if(write){
      print('saving sensitivity/specificity')
      write.csv(metricsTable, file.path(dir_model, paste0('metrics_', modelname, '.csv')))
      
      ggsave(
        filename = file.path(dir_model, paste0('PLOT_metrics_MODEL_', modelname,'.svg')),
        plot = plot_metrics,
        width = 6,
        height = 4
      )
    }
  
  print(paste0('done inspecting ', modelname))
  return(confusion_table)
}

modelnames <- list.files('./models') %>% 
  .[!.%in%c('archive', 'inspection')]

full <- mclapply(
  modelnames,
  inspect_model,
  write = T,
  mc.cores=cpus
)

names(full) <- modelnames

confusions <- lapply(
  modelnames,
  function(m){
    full[[m]][[2]] %>% 
      as.data.frame() %>% 
      mutate(
        model = m
      )
  }
) %>% 
  bind_rows() %>% 
  mutate(
    weight = str_extract(model, '(.*)Weight_', group =1),
    metadata = str_extract(model, '_(.*)Meta', group = 1)
  )

write.csv(confusions, file.path(dir_out, 'confusion_all.csv'), row.names = F)
