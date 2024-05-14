library(dplyr)
library(parallel)
library(stringr)
library(tidyr)
library(ggplot2)
library(caret)
library(shadowtext)

cpus <- 8


# currently using a prediction threshold of >0
inspect_model <- function(modelname, prediction_threshold=-1){
  print(paste0('LAUNCHING inspection of model ', modelname))
  
  #
  #  filesystem prep ----
  #
    dir_model <- file.path('models', modelname)
    dir_testing <- file.path(dir_model, 'tests')
    
    if(dir.exists(dir_testing)){
      print('model already analyzed; skipping')
      return(NULL)
    } else {dir.create(dir_testing)}
    
  #
  # Data read ----
  #

    data <- read.csv(file.path(dir_model, 'output_testfold.csv'))
    names(data) <- str_remove(names(data), 'score_')
    classes <- names(data) %>% 
      {.[!.%in%c('classes_actual', 'class_max')]}
    
    classes_buzz <- classes %>% 
      .[str_detect(., 'buzz')]
    
    data_long <- data %>% 
      select(!class_max) %>% 
      pivot_longer(
        cols = !classes_actual,
        names_to = 'neuron',
        values_to = 'activation'
      ) %>% 
      mutate(
        correct = str_detect(classes_actual, neuron)
      )
    
    data_predicted <- data %>% 
      select(!class_max) %>% 
      mutate(
        across(
          !classes_actual,
          ~.x > prediction_threshold
        )
      )
    
    weights <- read.csv(
      file.path(dir_model, 'weights.csv')
    )
    
  #
  # Confusion ----
  #
    # remember: confusion isn't so important anymore; we're calling based on numbers, not max
    # there's not a way to convert multi-event to confusion because predicting X given Y is not necessarily an error
    confusion <- data %>%
      filter(!str_detect(classes_actual, ';')) %>%  # drop multi-event samples; don't work in confusion table
      {caret::confusionMatrix(
        data = factor(.$class_max, levels = classes),
        reference = factor(.$classes_actual, levels = classes)
      )}
      
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
        data = expand.grid(foo = classes_buzz, bar = classes_buzz, proportion = 1),
        mapping = aes(x = bar, y = foo),
        fill = "#00000000",
        linewidth = 1.3,
        color = "#212121"
      ) #+
      # true buzz highlight; turned off because some models use common buzz at the moment
      # geom_tile(
      #   aes(x = 'ins_buzz_bee', y = 'ins_buzz_bee'),
      #   fill = "#00000000",
      #   linewidth = 1.3,
      #   color = "darkgray"
      # )
 
      write.csv(confusion_table, file.path(dir_testing, 'confusion.csv'), row.names = F)
      
      ggsave(
        filename = file.path(dir_testing, 'confusion.svg'),
        plot = plot_confusion,
        width = 12,
        height = 9
      )
      
  # TODO: also attach weights; do I need to revise this for multi-class classification?
  # Affinity ----
  #
    warning('column \'samples\' in affinities is currently calculated only from single classes')  # TODO: fix this; should be generated from training data somehow, I guess in train.py
    affinities <- confusion_table %>% 
      group_by(predicted) %>% 
      summarize(affinity = sum(proportion, na.rm = T), samples = sum(count)) %>% 
      rename('class' = predicted) %>% 
      left_join(weights, by = 'class')
   
    write.csv(affinities, file.path(dir_testing, 'affinities.csv'), row.names = F)
  
  #
  # Sensitivity, specificity ----
  #
    
    get_sensitivity <- function(class_in){
      # sensitivity = p(positive predicted | positive actual)
      sub_positive <- data_predicted 
      
      # odd step for the case of multiple classes in
      sub_positive$true <- sapply(
        sub_positive$classes_actual,
        function(c){T %in% str_detect(c, class_in)}
      )
      
      sub_positive <- filter(sub_positive, true) %>% 
        select(all_of(class_in))
      
      sub_positive$predicted <- apply(sub_positive, 1, function(r){T %in% r})
         
      
      count_total <- nrow(sub_positive)
      count_positive <- sum(sub_positive$predicted)
      
      sensitivity <- count_positive/count_total
      
      return(sensitivity)
    }
    
    get_specificity <- function(class_in){
      # specificity = p(negative predicted | negative actual)
      sub_negative <- data_predicted 
      
      # odd step for the case of multiple classes in
      sub_negative$true <- sapply(
        sub_negative$classes_actual,
        function(c){T %in% str_detect(c, class_in)}
      )
      
      sub_negative <- filter(sub_negative, !true) %>% 
        select(all_of(class_in))
      
      sub_negative$predicted <- apply(sub_negative, 1, function(r){T %in% r})
      
      count_total <- nrow(sub_negative)
      count_negative <- sum(!sub_negative$predicted)
      
      specificity <- count_negative/count_total
      
      return(specificity)
    }
    
    classes_common <- c('ambient_day', 'ambient_scraping', 'ambient_rustle', 'ambient_sound', 'mech_auto', 'mech_plane')
    get_specificity_common <- function(class_in){
      # here I want to make a more critical measure of specificity; take the
      # confusion of only the most prevalent classes. The model does great on rain and geese,
      # so what? They're rare. Let's look at the really common classes
      
      sub_negative <- data_predicted 
      
      # odd step for the case of multiple classes in
      sub_negative$true <- sapply(
        sub_negative$classes_actual,
        function(c){T %in% str_detect(c, class_in)}
      )
      
      sub_negative$common <- sapply(
        sub_negative$classes_actual,
        function(c){T %in% str_detect(c, classes_common)}
      )
      
      sub_negative <- filter(sub_negative, !true, common) %>% 
        select(all_of(class_in))
      
      sub_negative$predicted <- apply(sub_negative, 1, function(r){T %in% r})
      
      count_total <- nrow(sub_negative)
      count_negative <- sum(!sub_negative$predicted)
      
      specificity <- count_negative/count_total
      
      return(specificity)
    }
    
  #
  # Metrics ----
  #
    metricsTable <- data.frame(
      class = classes_buzz
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
          sensitivity = get_sensitivity(classes_buzz),
          specificity = get_specificity(classes_buzz),
          specificity_common = get_specificity_common(classes_buzz)
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
        label = round(value, 3) %>% 
          format(nsmall=3)
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
   
   write.csv(metricsTable, file.path(dir_testing, 'metrics.csv'), row.names = F)  # training metrics will be metrics_train
   
   ggsave(
     filename = file.path(dir_testing, 'metrics.svg'),
     plot = plot_metrics,
     width = 6,
     height = 4
   )
    
  #
  # Activations ----
  #
   n_col = max(ceiling(sqrt(length(classes))), 4)
   n_row = ceiling(length(classes)/n_col)
   
   plot_activations <- ggplot(
     data_long,
     aes(
       x = activation,
       color = correct
     )
   ) +
     geom_density() +
     geom_vline(xintercept=prediction_threshold, color='darkred') +
     facet_wrap(
       facets = vars(neuron),
       scales = 'free',
       ncol = n_col,
       nrow = n_row
     ) +
     ggtitle(paste0('activations for ', modelname))
   
   ggsave(
     filename = file.path(dir_testing, 'activations.svg'),
     plot = plot_activations,
     width = 3*n_col,
     height = 3*n_row
   )

  print(paste0('done inspecting ', modelname))
}

modelnames <- list.files('./models') %>% 
  .[!.%in%c('archive', 'test')] %>% 
  .[sapply(
    .,
    function(name){!file.exists(file.path('./models', name, 'tests'))}
  )]

full <- mclapply(
  modelnames,
  inspect_model,
  mc.cores=cpus
)
