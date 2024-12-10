library(tidyverse)
library(caret)
library(xgboost)
library(ggplot2)
library(scales)
library(corrplot)
library(usmap)
library(gridExtra)
library(skimr)
library(data.table)
library(scales)
library(parallel)
library(doParallel)
library(tictoc)
library(pROC)
library(furrr)

set.seed(506)

# import data
# medicare 
med20 <- read_csv('data/MUP_PHY_R22_P05_V10_D20_Prov_Svc.csv')
# med21 <- read_csv('data/MUP_PHY_R23_P05_V10_D21_Prov_Svc.csv')
# tax
tax20 <- read_csv("data/20zpallagi.csv")
# tax21 <- read_csv("data/21zpallagi.csv")

# use data in Year 2020
# filter out "cardio" related provider
# cardio_med20 <- med20 %>% filter(grepl("^cardiology$",tolower(Rndrng_Prvdr_Type)))

ped_med20 <- med20 %>% filter(grepl("pediatric medicine",tolower(Rndrng_Prvdr_Type)))

# statistic description
# check and clean NaN or abnormal data
#' Calculate summary statistics for numeric columns in a dataframe
#'
#' @param df A dataframe to analyze
#' @return A dataframe containing summary statistics for each numeric column:
#'   missing values, zeros, negative values, mean, median, and standard deviation
#' @examples
#' summary_stats(my_dataframe)
summary_stats <- function(df) {
  df %>% 
    summarise(across(where(is.numeric),
                     list(
                       missing = ~sum(is.na(.)),
                       zeros = ~sum(. == 0, na.rm = TRUE),
                       negative = ~sum(. < 0, na.rm = TRUE),
                       mean = ~mean(., na.rm = TRUE),
                       median = ~median(., na.rm = TRUE),
                       sd = ~sd(., na.rm = TRUE)
                     )))
}

medicare_stats <- summary_stats(ped_med20)
tax_stats <- summary_stats(tax20)

# clean data
clean_ped_medicare <- ped_med20 %>%
  filter(
    !is.na(Avg_Mdcr_Pymt_Amt),
    !is.na(Rndrng_Prvdr_State_FIPS),
    Avg_Mdcr_Pymt_Amt >= 0,
    Tot_Srvcs > 0,
    Tot_Benes > 0
  )

clean_tax20 <- tax20 %>% 
  filter(!is.na(STATEFIPS))

clean_ped_medicare <- clean_ped_medicare %>% 
  mutate(Rndrng_Prvdr_Crdntls = if_else(is.na(Rndrng_Prvdr_Crdntls), "O", Rndrng_Prvdr_Crdntls),
         Rndrng_Prvdr_Gndr = if_else(is.na(Rndrng_Prvdr_Gndr), "O", Rndrng_Prvdr_Gndr))

n_cores <- availableCores() - 1  
plan(multisession, workers = n_cores)

tic("Parallel aggregation with furrr")
clean_tax20_new <- clean_tax20 %>% 
  select(STATEFIPS, zipcode, starts_with("A"), N2) %>% 
  group_by(STATEFIPS, zipcode) %>% 
  summarize(across(
    starts_with("A"),
    ~ weighted.mean(., weight = N2)
  ), .groups = "drop") %>% 
  select(-agi_stub)
toc()

#' Calculate weighted mode for a vector
#'
#' @param x Vector of values
#' @param w Vector of weights
#' @return The weighted mode of x
#' @examples
#' weighted_mode(c(1,2,2,3), c(1,1,2,1))
weighted_mode <- function(x, w) {
  if(length(x) == 0 || length(w) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tapply(w, x, sum))]
}

aggregated_tax <- clean_tax20 %>%
  group_by(STATEFIPS, zipcode) %>%
  summarise(
    # 使用N2作为权重计算agi_stub的加权众数
    agi_stub = weighted_mode(agi_stub, N2),
    .groups = "drop"
  )

clean_tax20_new <- clean_tax20_new %>% 
  left_join(aggregated_tax, by = c("zipcode","STATEFIPS"))

# merge two dataset and select possible features
merged_data <- clean_ped_medicare %>% 
  # join on the key "zipcode"
  left_join(clean_tax20_new, 
            by = c("Rndrng_Prvdr_Zip5" = "zipcode",
                   "Rndrng_Prvdr_State_FIPS" = "STATEFIPS"),
            keep = TRUE) %>% 
  select(
    # medicare payment
    Avg_Mdcr_Pymt_Amt,
    Avg_Sbmtd_Chrg,
    
    # medicare provider related features
    Rndrng_Prvdr_Ent_Cd,
    Rndrng_Prvdr_Gndr,
    Rndrng_Prvdr_Crdntls,
    Rndrng_Prvdr_Type,
    Rndrng_Prvdr_Mdcr_Prtcptg_Ind,
    
    # medicare services information
    HCPCS_Cd,
    HCPCS_Drug_Ind,
    Tot_Benes,
    Tot_Srvcs,
    Tot_Bene_Day_Srvcs,
    Place_Of_Srvc,
    
    # geography and tax related features
    STATEFIPS,
    # zipcode,
    agi_stub,
    A00100,
    A00200,
    A00300,
    A00700,
    A01700,
    A02500,
    A17000,
    A19700,
    A85530,
    # N03220,
    A03220,
    # N03210,
    A03210,
    # N07180,
    A07180,
    # N07225,
    A07225,
    # N11070,
    A11070,
    # N11450,
    A11450
  ) %>% na.omit()

model_data <- merged_data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric)) %>% 
  na.omit()

model_matrix <- model.matrix(~ . - 1, data = model_data %>% select(-Avg_Mdcr_Pymt_Amt))

tic("Total Runtime")

# # check GPU support
# cat("Checking GPU support...\n")
# has_gpu <- tryCatch({
#   data <- matrix(1:10, ncol=1)
#   label <- c(0,1,0,1,0,1,0,1,0,1)
#   test <- xgb.DMatrix(data, label=label)
#   params <- list(tree_method="gpu_hist")
#   bst <- xgb.train(params, test, nrounds=1)
#   TRUE
# }, error = function(e) {
#   FALSE
# })
# 
# if(has_gpu) {
#   cat("GPU support detected, using GPU for training...\n")
#   tree_method <- "gpu_hist"
# } else {
#   cat("No GPU support detected, using CPU...\n")
#   tree_method <- "hist"
# }

# set up the parameter ranges for the param search
cat("Setting up parameter ranges...\n")
param_ranges <- list(
  max_depth = 3:10,
  eta = c(0.01, 0.05, 0.1, 0.15, 0.2, 0.3),
  gamma = seq(0, 0.5, 0.1),
  colsample_bytree = seq(0.6, 1.0, 0.1),
  min_child_weight = 1:5,
  subsample = seq(0.6, 1.0, 0.1),
  nrounds = seq(100, 500, 100)
)

# Generate random parameters combination
n_combinations <- 70  # Increased to 70 combinations
set.seed(506)  

# randomly generated the parameter combinations
random_params <- data.frame(
  max_depth = sample(param_ranges$max_depth, n_combinations, replace = TRUE),
  eta = sample(param_ranges$eta, n_combinations, replace = TRUE),
  gamma = sample(param_ranges$gamma, n_combinations, replace = TRUE),
  colsample_bytree = sample(param_ranges$colsample_bytree, n_combinations, replace = TRUE),
  min_child_weight = sample(param_ranges$min_child_weight, n_combinations, replace = TRUE),
  subsample = sample(param_ranges$subsample, n_combinations, replace = TRUE),
  nrounds = sample(param_ranges$nrounds, n_combinations, replace = TRUE)
)

# print the information of each parameter combination
cat(sprintf("Generated %d random parameter combinations\n", n_combinations))
cat("Parameter combinations preview:\n")
print(head(random_params, 3))
cat("...\n\n")

# set up cross validation
cv_folds <- 5
cat(sprintf("Using %d-fold cross-validation\n", cv_folds))

# prepare data matrix
dtrain <- xgb.DMatrix(model_matrix, label = model_data$Avg_Mdcr_Pymt_Amt)

# initialize the result storage
results <- data.frame()
best_rmse <- Inf
best_params <- NULL

# Start random search
cat("\nStarting random search...\n")
tic("Random Search")

for(i in 1:nrow(random_params)) {
  # current parameter search
  params <- list(
    max_depth = random_params$max_depth[i],
    eta = random_params$eta[i],
    gamma = random_params$gamma[i],
    colsample_bytree = random_params$colsample_bytree[i],
    min_child_weight = random_params$min_child_weight[i],
    subsample = random_params$subsample[i],
    tree_method = 'hist',
    objective = "reg:squarederror"
  )
  
  # to show the progress
  cat(sprintf("\nTesting combination %d/%d (%.1f%%)\n", 
              i, n_combinations, i/n_combinations*100))
  cat("Parameters:", paste(names(params), unlist(params), sep="=", collapse=" "))
  
  # perfrom cross validation
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = random_params$nrounds[i],
    nfold = cv_folds,
    early_stopping_rounds = 20,  # increase early stopping rounds
    verbose = FALSE,
    metrics = "rmse"
  )
  
  # record results
  current_rmse <- min(cv_results$evaluation_log$test_rmse_mean)
  results <- rbind(results, 
                   data.frame(params = I(list(params)), 
                              nrounds = random_params$nrounds[i],
                              rmse = current_rmse))
  
  # update the best parameter combination
  if(current_rmse < best_rmse) {
    best_rmse <- current_rmse
    best_params <- c(params, nrounds = random_params$nrounds[i])
    cat("\nNew best RMSE:", best_rmse, "\n")
  }
}


grid_search_time <- toc(log = TRUE)

# use the model with the best parameter combination
cat("\nTraining final model with best parameters...\n")
tic("Final Model Training")
final_xgb <- xgboost(
  params = best_params,
  data = dtrain,
  nrounds = best_params$nrounds,
  verbose = 1
)
final_model_time <- toc(log = TRUE)

# list the top 20 importance features
cat("\nCalculating feature importance...\n")
importance_matrix <- xgb.importance(model = final_xgb)
cat("\nTop 20 most important features:\n")
print(head(importance_matrix, 20))

# save the result
results_summary <- list(
  best_params = best_params,
  best_rmse = best_rmse,
  importance_matrix = importance_matrix,
  grid_search_results = results,
  grid_search_time = grid_search_time$toc - grid_search_time$tic,
  final_model_time = final_model_time$toc - final_model_time$tic
)

# print the running time
total_time <- toc(log = TRUE)

# run time summary
cat("\n=== Runtime Summary ===\n")
cat("Grid Search Time:", grid_search_time$toc - grid_search_time$tic, "seconds\n")
cat("Final Model Training:", final_model_time$toc - final_model_time$tic, "seconds\n")
cat("Total Runtime:", total_time$toc - total_time$tic, "seconds\n")

# save the result
save(results_summary, file = "xgboost_grid_search_results.RData")
cat("\nResults saved to 'xgboost_grid_search_results.RData'\n")

importance_plot <- ggplot(head(importance_matrix, 20), 
                          aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # horizontal bar plot
  labs(title = "Top 20 Feature Importance",
       x = "Features",
       y = "Gain") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    panel.grid.major.y = element_blank()
  )

# save the figure
ggsave("feature_importance.png", importance_plot, 
       width = 12, height = 8, dpi = 300)


# importance analysis from XGBoost Model
importance_matrix <- xgb.importance(model = final_xgb)
top_features <- head(importance_matrix$Feature, 20)  # choose the top 20 features

# define the high payment using 75% quantile
threshold <- quantile(model_data$Avg_Mdcr_Pymt_Amt, 0.75)  
model_data$high_payment <- ifelse(model_data$Avg_Mdcr_Pymt_Amt > threshold, 1, 0)

#' Analyze the impact of a feature on high payment probability
#'
#' @param feature Name of the feature to analyze
#' @param data Dataset containing the feature and payment information
#' @return List containing model results, AUC value, and feature type
#' @examples
#' analyze_feature_impact("Tot_Srvcs", model_data)
analyze_feature_impact <- function(feature, data) {
  cat(sprintf("\n=== Analyzing feature: %s ===\n", feature))
  
  # validate and sanitize the data
  if(sum(!is.na(data[[feature]])) == 0) {
    cat("Feature contains only NA values\n")
    return(NULL)
  }
  
  if(length(unique(data[[feature]][!is.na(data[[feature]])])) < 2) {
    cat("Feature contains less than 2 unique values\n")
    return(NULL)
  }
  
  cat("Data check:\n")
  cat("- Class:", class(data[[feature]]), "\n")
  cat("- NA count:", sum(is.na(data[[feature]])), "\n")
  cat("- Unique values:", length(unique(data[[feature]])), "\n")
  
  # Process feature based on its type
  if(feature == "agi_stub") {
    feature_to_use <- feature
  } else if(feature %in% c("STATEFIPS", "Rndrng_Prvdr_State_FIPS")) {
    feature_to_use <- feature
  } else if(feature %in% c("Rndrng_Prvdr_Ent_Cd", "Rndrng_Prvdr_Gndr", "Rndrng_Prvdr_Crdntls", 
                           "Rndrng_Prvdr_Type", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind", "HCPCS_Cd", 
                           "HCPCS_Drug_Ind", "Place_Of_Srvc")) {
    feature_to_use <- feature
  } else if(grepl("^Tot_", feature) || grepl("^Avg_", feature) || 
            grepl("^[A][0-9]{5}", feature) || grepl("^[N][0-9]{5}", feature)) {
    # Process continuous variables
    # Add data validation
    valid_data <- data[[feature]][!is.na(data[[feature]])]
    if(length(valid_data) < 5) {
      cat("Not enough valid data points\n")
      return(NULL)
    }
    
    # 1. Ensure enough unique values
    unique_vals <- unique(valid_data)
    if(length(unique_vals) >= 5) {
      # 2. Calculate quantiles
      breaks <- try({
        unique(quantile(valid_data, probs = seq(0, 1, 0.2)))
      }, silent = TRUE)
      
      # 3. If quantile calculation fails or insufficient breaks, use equal-width bins
      if(inherits(breaks, "try-error") || length(breaks) < 5) {
        min_val <- min(valid_data)
        max_val <- max(valid_data)
        if(min_val != max_val) {
          breaks <- seq(min_val, max_val, length.out = 6)
        } else {
          feature_to_use <- feature
        }
      }
      
      if(exists("breaks") && length(unique(breaks)) >= 5) {
        data[[paste0(feature, "_bin")]] <- cut(data[[feature]], 
                                               breaks = breaks,
                                               labels = c("Very Low", "Low", "Medium", 
                                                          "High", "Very High"),
                                               include.lowest = TRUE)
        feature_to_use <- paste0(feature, "_bin")
      } else {
        feature_to_use <- feature
      }
    } else {
      feature_to_use <- feature
    }
  } else {
    feature_to_use <- feature
  }
  
  # delete Null values
  valid_rows <- !is.na(data[[feature_to_use]]) & !is.na(data$high_payment)
  if(sum(valid_rows) < 5) {
    cat("Not enough valid rows after NA removal\n")
    return(NULL)
  }
  
  model_data <- data[valid_rows, ]
  
  # Ensure sufficient variation in target variable
  if(length(unique(model_data$high_payment)) < 2) {
    cat("Target variable has insufficient variation\n")
    return(NULL)
  }
  
  # logistic regression
  tryCatch({
    formula <- as.formula(paste("high_payment ~", feature_to_use))
    model <- glm(formula, data = model_data, family = binomial())
    
    # calculate predicted values and AUC
    predictions <- predict(model, type = "response")
    
    if(length(predictions) != length(model_data$high_payment)) {
      cat("Prediction length mismatch\n")
      return(NULL)
    }
    
    roc_obj <- roc(model_data$high_payment, predictions)
    cat("Analysis successful\n")
    cat("AUC value:", auc(roc_obj), "\n")
    
    return(list(
      model = model,
      auc = auc(roc_obj),
      summary = summary(model),
      feature_type = if(feature_to_use == feature) "original" else "binned"
    ))
  }, error = function(e) {
    cat("Error in model fitting:", e$message, "\n")
    return(NULL)
  })
}

#' Create visualization for feature impact analysis
#'
#' @param feature Name of the feature to visualize
#' @param data Dataset containing the feature and payment information
#' @return ggplot object showing the feature's impact on high payment probability
#' @examples
#' plot_feature_impact("Tot_Srvcs", model_data)
plot_feature_impact <- function(feature, data) {
  if(feature == "AGI_STUB") {
    feature_to_plot <- feature
    xlabel <- "Income Level"
  } else if(feature %in% c("STATEFIPS", "Rndrng_Prvdr_State_FIPS")) {
    feature_to_plot <- feature
    xlabel <- "State"
  } else if(grepl("^Tot_", feature) || grepl("^Avg_", feature) || 
            grepl("^[A][0-9]{5}", feature) || grepl("^[N][0-9]{5}",feature)) {
    
    # check the data size
    unique_vals <- unique(data[[feature]])
    n_unique <- length(unique_vals)
    
    if(n_unique >= 5) {
      breaks <- try({
        unique(quantile(data[[feature]], probs = seq(0, 1, 0.2)))
      }, silent = TRUE)
      if(inherits(breaks, "try-error") || length(breaks) < 5) {
        min_val <- min(data[[feature]])
        max_val <- max(data[[feature]])
        if(min_val == max_val) {
          feature_to_plot <- feature
          xlabel <- gsub("_", " ", feature)
        } else {
          breaks <- seq(min_val, max_val, length.out = 6)
          if(length(unique(breaks)) < 5) {
            feature_to_plot <- feature
            xlabel <- gsub("_", " ", feature)
          } else {
            data[[paste0(feature, "_bin")]] <- cut(data[[feature]], 
                                                   breaks = breaks,
                                                   labels = c("Very Low", "Low", "Medium", 
                                                              "High", "Very High"),
                                                   include.lowest = TRUE)
            feature_to_plot <- paste0(feature, "_bin")
            xlabel <- gsub("_", " ", feature)
          }
        }
      } else {
        data[[paste0(feature, "_bin")]] <- cut(data[[feature]], 
                                               breaks = breaks,
                                               labels = c("Very Low", "Low", "Medium", 
                                                          "High", "Very High"),
                                               include.lowest = TRUE)
        feature_to_plot <- paste0(feature, "_bin")
        xlabel <- gsub("_", " ", feature)
      }
    } else {
      feature_to_plot <- feature
      xlabel <- gsub("_", " ", feature)
    }
  } else {
    feature_to_plot <- feature
    xlabel <- gsub("_", " ", feature)
  }
  
  # calculate the high payment proportion in each group of each feature
  impact_summary <- data %>%
    group_by(!!sym(feature_to_plot)) %>%
    summarise(
      high_payment_prop = mean(high_payment),
      n = n(),
      se = sqrt((high_payment_prop * (1-high_payment_prop))/n)
    )
  
  # plot
  p <- ggplot(impact_summary, aes_string(x = feature_to_plot, y = "high_payment_prop")) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = high_payment_prop - se, 
                      ymax = high_payment_prop + se), 
                  width = 0.2) +
    labs(title = paste("Impact of", xlabel, "on High Payment Probability"),
         y = "Probability of High Payment",
         x = xlabel) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # to constrain only show the top 10 countries
  if(feature %in% c("STATEFIPS", "Rndrng_Prvdr_State_FIPS")) {
    p <- p + scale_x_discrete(breaks = impact_summary %>% 
                                top_n(10, high_payment_prop) %>% 
                                pull(!!sym(feature_to_plot)))
  }
  
  return(p)
}

# apply feature impact analysis
feature_impacts <- lapply(top_features, function(feature) {
  tryCatch({
    analyze_feature_impact(feature, model_data)
  }, error = function(e) {
    cat("Error analyzing feature:", feature, "\n")
    cat("Error message:", e$message, "\n")
    return(NULL)
  })
})

# plot
impact_plots <- lapply(top_features, function(feature) {
  tryCatch({
    plot_feature_impact(feature, model_data)
  }, error = function(e) {
    cat("Error plotting feature:", feature, "\n")
    cat("Error message:", e$message, "\n")
    return(NULL)
  })
})

# remove the NULL results
impact_plots <- impact_plots[!sapply(impact_plots, is.null)]

# combine and save to a pdf file
pdf("feature_impacts.pdf", width = 12, height = 8)
for(i in seq(1, length(impact_plots), 2)) {
  if(i + 1 <= length(impact_plots)) {
    grid.arrange(impact_plots[[i]], impact_plots[[i+1]], ncol = 2)
  } else {
    grid.arrange(impact_plots[[i]], ncol = 1)
  }
}
dev.off()


valid_impacts <- feature_impacts[!sapply(feature_impacts, is.null)]

# initialize the summary data fame
impact_summary <- data.frame(
  Feature = top_features,  
  AUC = NA,               
  Feature_Type = NA       
)

# fill in the data frame
for(i in seq_along(valid_impacts)) {
  if(!is.null(valid_impacts[[i]])) {
    feature_name <- top_features[i]
    impact_summary$AUC[impact_summary$Feature == feature_name] <- valid_impacts[[i]]$auc
    impact_summary$Feature_Type[impact_summary$Feature == feature_name] <- valid_impacts[[i]]$feature_type
  }
}

# remove NULL
impact_summary <- impact_summary %>%
  filter(!is.na(AUC))

# order by AUC descending
impact_summary <- impact_summary %>%
  arrange(desc(AUC))

print(impact_summary)
save(impact_summary, feature_impacts, impact_plots, 
     file = "feature_analysis_results.RData")

#' Analyze features within a specific income group
#'
#' @param data Complete dataset
#' @param income_levels Vector of income levels to include
#' @param group_name Name of the income group for reporting
#' @return List containing analysis summary, impacts, and plots
#' @examples
#' analyze_income_group(model_data, c(1,2), "Low")
analyze_income_group <- function(data, income_levels, group_name) {
  
  cat(sprintf("\nAnalyzing %s income group:\n", group_name))
  cat("Income levels included:", paste(income_levels, collapse=", "), "\n")
  
  # Filter data for income group
  group_data <- data %>%
    filter(agi_stub %in% income_levels)
  
  # to check if there is enough data
  if(nrow(group_data) == 0) {
    cat(sprintf("No data available for %s income group\n", group_name))
    return(NULL)
  }
  
  cat(sprintf("Number of records in group: %d\n", nrow(group_data)))
  cat("Distribution of agi_stub in group:\n")
  print(table(group_data$agi_stub))
  
  # to check if there is enough data
  if(nrow(group_data) <= 0) { 
    cat(sprintf("Insufficient data for %s income group (less than 10 records)\n", group_name))
    return(NULL)
  }
  
  # to check the null value
  valid_features <- sapply(top_features, function(feature) {
    sum(!is.na(group_data[[feature]])) > 0
  })
  
  features_to_analyze <- names(valid_features)[valid_features]
  
  if(length(features_to_analyze) == 0) {
    cat(sprintf("No valid features for %s income group\n", group_name))
    return(NULL)
  }
  
  # Define high payment threshold for this group
  threshold <- quantile(group_data$Avg_Mdcr_Pymt_Amt, 0.75, na.rm = TRUE)
  group_data$high_payment <- ifelse(group_data$Avg_Mdcr_Pymt_Amt > threshold, 1, 0)
  
  # Get feature impacts only for valid features
  feature_impacts <- lapply(features_to_analyze, function(feature) {
    tryCatch({
      analyze_feature_impact(feature, group_data)
    }, error = function(e) {
      cat(sprintf("\nError analyzing feature %s in %s income group: %s\n", 
                  feature, group_name, e$message))
      return(NULL)
    })
  })
  names(feature_impacts) <- features_to_analyze
  
  # Generate visualizations only for valid features
  impact_plots <- lapply(features_to_analyze, function(feature) {
    tryCatch({
      p <- plot_feature_impact(feature, group_data)
      p <- p + ggtitle(paste(group_name, "Income Group:", 
                             p$labels$title))
      return(p)
    }, error = function(e) {
      cat(sprintf("\nError plotting feature %s in %s income group: %s\n",
                  feature, group_name, e$message))
      return(NULL)
    })
  })
  names(impact_plots) <- features_to_analyze
  
  # Save plots only if we have valid plots
  valid_plots <- impact_plots[!sapply(impact_plots, is.null)]
  if(length(valid_plots) > 0) {
    pdf(sprintf("feature_impacts_%s_income.pdf", tolower(group_name)), 
        width = 12, height = 8)
    for(i in seq(1, length(valid_plots), 2)) {
      if(i + 1 <= length(valid_plots)) {
        grid.arrange(valid_plots[[i]], valid_plots[[i+1]], ncol = 2)
      } else {
        grid.arrange(valid_plots[[i]], ncol = 1)
      }
    }
    dev.off()
  }
  
  # Summarize results
  valid_impacts <- feature_impacts[!sapply(feature_impacts, is.null)]
  impact_summary <- data.frame(
    Income_Group = group_name,
    Feature = features_to_analyze,  
    AUC = NA,
    Feature_Type = NA
  )
  
  for(i in seq_along(valid_impacts)) {
    if(!is.null(valid_impacts[[i]])) {
      feature_name <- features_to_analyze[i]
      impact_summary$AUC[impact_summary$Feature == feature_name] <- 
        valid_impacts[[i]]$auc
      impact_summary$Feature_Type[impact_summary$Feature == feature_name] <- 
        valid_impacts[[i]]$feature_type
    }
  }
  
  impact_summary <- impact_summary %>%
    filter(!is.na(AUC)) %>%
    arrange(desc(AUC))
  
  list(
    summary = impact_summary,
    impacts = valid_impacts,
    plots = valid_plots
  )
}
# Analyze each income group
results <- list(
  low = analyze_income_group(model_data, c(1,2), "Low"),
  medium = analyze_income_group(model_data, c(3,4), "Medium"), 
  high = analyze_income_group(model_data, c(5,6), "High")
)

for(group in c("low", "medium", "high")) {
  if(is.null(results[[group]])) {
    cat(sprintf("\nWarning: No valid results for %s income group\n", group))
  }
}

# Compare results across groups
all_summaries <- bind_rows(
  results$low$summary,
  results$medium$summary,
  results$high$summary
)

# Save results
save(results, all_summaries, file = "income_group_analysis.RData")

# Print comparison
print("=== Feature Impacts by Income Group ===")
print(all_summaries %>% 
        select(Income_Group, Feature, AUC) %>%
        spread(Income_Group, AUC) %>%
        arrange(desc(Low)))