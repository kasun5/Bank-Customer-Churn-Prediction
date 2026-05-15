
############################################################
# Bank Customer Churn Prediction
# 
# Description: End-to-end machine learning pipeline for predicting customer 
#              churn using logistic regression with multiple sampling techniques
############################################################


############################
# 1. SETUP & LIBRARIES
############################

# Required packages
required_packages <- c(
  "dplyr",        # Data manipulation
  "ggplot2",      # Visualization
  "patchwork",    # Combining plots
  "here",         # File path management
  "caret",        # ML utilities
  "caTools",      # Train/test split
  "smotefamily",  # SMOTE implementation
  "ggcorrplot",   # Correlation visualization
  "tibble",       # Modern data frames
  "pROC",         # ROC curve and AUC calculation
  "scales"        # Additional plot utilities
)

# Install missing packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) install.packages(new_packages)
}

install_if_missing(required_packages)

# Load libraries
invisible(lapply(required_packages, library, character.only = TRUE))

# Set random seed for reproducibility
set.seed(123)

# Create output directories
dir.create("plots", showWarnings = FALSE, recursive = TRUE)
dir.create("models", showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)


####################################
# 2. DATA LOADING & INITIAL CLEANING
####################################

message("Loading data")

# load data
bank_raw <- read.csv(here("data", "Bank_Customer_Churn_Data.csv"), stringsAsFactors = FALSE)

cat("\n Initial Check \n")
# quick check for correct data types and column names
str(bank_raw)

message("\n Cleaning and preprocessing data")
# remove customer_id column 
bank_raw <- bank_raw %>% select(-customer_id)

# Convert target to factor for classification
bank_raw$churn <- as.factor(bank_raw$churn)

# Convert products_number, credit_card, active_member to factors
bank_raw <- bank_raw %>%
  mutate(across(c(products_number, credit_card, active_member), as.factor))


# Check total missing values per column
cat("\n Missing Values Summary \n")
colSums(is.na(bank_raw))

# summary of the data
summary(bank_raw)

# Separate numerical and categorical features
numeric_columns <- bank_raw %>%
  select(where(is.numeric)) %>%
  colnames()

categorical_columns <- bank_raw %>%
  select(where(~ is.factor(.x) | is.character(.x))) %>%
  colnames()


####################################
# 3. EXPLORATORY DATA ANALYSIS (EDA)
####################################

message("\nGenerating EDA visualizations")

# 3.1 Generate numeric Variable Distributions
hist_list <- lapply(numeric_columns, function(col) {
  ggplot(bank_raw, aes(x = .data[[col]])) +  # Use .data[[col]] to dynamically reference column names
    geom_histogram(fill = "steelblue", color = "white", bins = 30, alpha = 0.8) +
    labs(title = paste("Distribution of", col), x = col, y = "Frequency") +
    theme_minimal()+
    theme(
      plot.title = element_text(face = "bold", size = 11),
      axis.title = element_text(size = 10)
    )
})
# Display all histograms in a grid
hist_grid <- wrap_plots(hist_list, ncol = 3) +
  plot_annotation(
    title = "Distribution of Numeric Features",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
  )
print(hist_grid)


# 3.2 Boxplot - Numeric Variables by Churn Status
# Generate a list of boxplots
boxplot_list <- lapply(numeric_columns, function(col) {
  ggplot(bank_raw, aes(x = churn, y = .data[[col]], fill = churn)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red", outlier.shape = 1) +
    labs(title = paste(col, "by Churn"), 
         x = "Churn Status", 
         y = col) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 11),
      axis.title = element_text(size = 10)
    )
})

# Display all boxplots in a grid
boxplot_grid <- wrap_plots(boxplot_list, ncol = 3) +
  plot_annotation(
    title = "Numeric Features by Churn Status",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
  )
print(boxplot_grid)


# 3.3 Categorical Variables by Churn Status
# Generate a list of bar charts by churn status
cat_plot_list <- lapply(categorical_columns, function(col) {
  ggplot(bank_raw, aes(x = .data[[col]], fill = churn)) +
    geom_bar(position = "dodge") +
    # Use after_stat(count) instead of ..count..
    geom_text(stat = 'count', 
              aes(label = after_stat(count)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, 
              size = 3) +
    labs(
      title = paste("Churn by", gsub("_", " ", tools::toTitleCase(col))),
      x = gsub("_", " ", tools::toTitleCase(col)),
      y = "Count",
      fill = "Churned"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 11),
      axis.title = element_text(size = 10)
    )

})

# Display all categorical plots in a grid
cat_grid <- wrap_plots(cat_plot_list, ncol = 2) +
  plot_annotation(
    title = "Categorical Features by Churn Status",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
  )
print(cat_grid)


#  3.4 Correlation Analysis
# Calculate correlation matrix
correlation_matrix <- cor(bank_raw[numeric_columns], use = "pairwise.complete.obs")

# Visualize correlation matrix with a heat map
corr_plot <- ggcorrplot(
  correlation_matrix,
  method = "circle",
  type = "lower",
  lab = TRUE,
  lab_size = 3,
  title = "Correlation Matrix of Numeric Variables",
  ggtheme = theme_minimal()
  )
print(corr_plot)

# Save all EDA plots
ggsave("plots/01_histogram_distributions.png", hist_grid, width = 12, height = 8, dpi = 300)
ggsave("plots/02_boxplots_by_churn.png", boxplot_grid, width = 12, height = 8, dpi = 300)
ggsave("plots/03_categorical_by_churn.png", cat_grid, width = 10, height = 6, dpi = 300)
ggsave("plots/04_correlation_matrix.png", corr_plot, width = 8, height = 6, dpi = 300)

message("EDA visualizations saved to 'plots' folder.")


############################
# 4. FEATURE ENGINEERING
############################
message("\nApplying feature engineering")


# Log Transformation of Age
bank_raw <- bank_raw %>%
  mutate(age_log = log(age + 0.00123))

# One-Hot Encoding
# Using '-1' to get a column for every category level
y <- factor(bank_raw$churn)
X <- model.matrix(~ . - churn - age, data = bank_raw)[, -1]  # Remove intercept

bank_data <- as.data.frame(X)
bank_data$churn <- y

message("\nFeature engineering complete")


############################
# 5. TRAIN / TEST SPLIT
############################


# Split the data into training and testing sets
split <- sample.split(bank_data$churn, SplitRatio = 0.70)
train_data <- subset(bank_data, split == TRUE)
test_data <- subset(bank_data, split == FALSE)

cat(sprintf("Training set: %d samples\n", nrow(train_data)))
cat(sprintf("Test set: %d samples\n", nrow(test_data)))

# Update numeric columns to include engineered features
numeric_columns <- c(numeric_columns[numeric_columns != "age"], "age_log")

message("Scaling numeric features...")
# Scale numeric columns
scaler_params <- preProcess(train_data[, numeric_columns], method = c("center", "scale"))

# Apply the transformation to both dataset
train_data[, numeric_columns] <- predict(scaler_params, train_data[, numeric_columns])
test_data[, numeric_columns] <- predict(scaler_params, test_data[, numeric_columns])

# Save scaler
saveRDS(scaler_params, "models/scaler_params.rds")
message("Scaler parameters saved for deployment.")



############################
# 6. HANDLING CLASS IMBALANCE
############################

message("\nAddressing class imbalance...")

# 6.1 Class Weights for Weighted Logistic Regression
w_total <- nrow(train_data)
w_min <- sum(train_data$churn == 1)
w_maj <- sum(train_data$churn == 0)
case_weights <- ifelse(train_data$churn == 1, 
                                  w_total / (2 * w_min), 
                                  w_total / (2 * w_maj))

cat(sprintf("Class weights - Majority: %.2f, Minority: %.2f\n",
            w_total / (2 * w_maj),
            w_total / (2 * w_min)))



# 7.2 Prepare Data for SMOTE (Synthetic Minority Oversampling)
message("Applying SMOTE...")

smote_output <- SMOTE(X = as.data.frame(train_data %>% select(-churn)), 
                      target = train_data$churn, K = 5, dup_size = 0)
train_smote <- smote_output$data %>% rename(churn = class) %>% 
                      mutate(churn = as.factor(churn))

cat(sprintf("SMOTE applied - New training size: %d samples\n", nrow(train_smote)))


# 7.3 Random Downsampling
message("Applying downsampling")

down_train <- downSample(x = train_data %>% select(-churn),
                         y = train_data$churn,
                         yname = "churn")

cat(sprintf("Downsampling applied - New training size: %d samples\n", 
            nrow(down_train)))


############################
# 7. MODEL TRAINING
############################
message("\nTraining logistic regression models...")


# 7.1 Baseline Model (No class balancing)
base_fit <- glm(churn ~ ., data = train_data, family = binomial(link = "logit"))

# 7.2 Weighted Model
weight_fit <- glm(churn ~ ., data = train_data, family = quasibinomial(link = "logit"), 
                  weights = case_weights)

# 7.3 SMOTE Model
smote_fit <- glm(churn ~ ., data = train_smote, family = binomial(link = "logit"))

# 7.4 Downsampled Model
down_fit <- glm(churn ~ ., data = down_train, family = binomial(link = "logit"))

# Save models
saveRDS(base_fit, "models/model_baseline.rds")
saveRDS(weight_fit, "models/model_weighted.rds")
saveRDS(smote_fit, "models/model_smote.rds")
saveRDS(down_fit, "models/model_downsampled.rds")

message("Models saved to 'models' folder.")

############################
# 8.  MODEL EVALUATION
############################

message("\nEvaluating models")

# Evaluation Function
evaluate_model <- function(model, data, model_name, threshold = 0.5, set_name = "Test") {
  
  # Generate predictions
  probs <- predict(model, newdata = data, type = "response")
  preds <- factor(ifelse(probs >= threshold, 1, 0), levels = c(0, 1))
  
  # Confusion matrix
  cm <- confusionMatrix(preds, data$churn, positive = "1")
  
  # AUC calculation (threshold-independent metric)
  y_true_num <- as.numeric(as.character(data$churn))
  roc_obj <- roc(y_true_num, probs, quiet = TRUE)
  auc_score <- as.numeric(auc(roc_obj))
  
  # Extract metrics
  tibble(
    Model = model_name,
    Set = set_name,
    Threshold = threshold,
    Accuracy = cm$overall["Accuracy"],
    AUC = auc_score,
    Sensitivity = cm$byClass["Sensitivity"],  # Recall / True Positive Rate
    #Specificity = cm$byClass["Specificity"],  # True Negative Rate
    Precision = cm$byClass["Pos Pred Value"],
    F1_Score = cm$byClass["F1"]
  )
}


# 8.1 Training Set Performance 
message("Evaluating on training set")

results_train <- bind_rows(
  evaluate_model(base_fit, train_data, "Baseline", set_name = "Train"),
  evaluate_model(weight_fit, train_data, "Weighted", set_name = "Train"),
  evaluate_model(smote_fit, train_smote, "SMOTE", set_name = "Train"),
  evaluate_model(down_fit, down_train, "Downsampled", set_name = "Train")
)

print(results_train, n = Inf)


# 8.2 Test Set Performance 
message("\nEvaluating on test set")

results_test <- bind_rows(
  evaluate_model(base_fit, test_data, "Baseline", set_name = "Test"),
  evaluate_model(weight_fit, test_data, "Weighted", set_name = "Test"),
  evaluate_model(smote_fit, test_data, "SMOTE", set_name = "Test"),
  evaluate_model(down_fit, test_data, "Downsampled", set_name = "Test")
)

print(results_test, n = Inf)

# 8.3 Combined Results 
results_combined <- bind_rows(results_train, results_test)

# Save results
write.csv(results_train, "results/model_comparison_train.csv", row.names = FALSE)
write.csv(results_test, "results/model_comparison_test.csv", row.names = FALSE)
write.csv(results_combined, "results/model_comparison_combined.csv", row.names = FALSE)

message("\n Results saved to 'results' folder.")

# 8.4 Check for Overfitting 
cat("\n Overfitting Analysis \n")

overfitting_check <- results_combined %>%
  select(Model, Set, Accuracy, AUC, F1_Score) %>%
  tidyr::pivot_wider(names_from = Set, values_from = c(Accuracy, AUC, F1_Score)) %>%
  mutate(
    Accuracy_Diff = Accuracy_Train - Accuracy_Test,
    AUC_Diff = AUC_Train - AUC_Test,
    F1_Diff = F1_Score_Train - F1_Score_Test
  )

print(overfitting_check)


# 8.5 Identify Best Model
best_model_name <- results_test %>%
  arrange(desc(F1_Score)) %>%
  slice(1) %>%
  pull(Model)

best_f1 <- results_test %>%
  filter(Model == best_model_name) %>%
  pull(F1_Score)

best_auc <- results_test %>%
  filter(Model == best_model_name) %>%
  pull(AUC)

message(sprintf("\n🏆 Best performing model: %s (F1: %.4f, AUC: %.4f)", 
                best_model_name, best_f1, best_auc))


############################
# 9. VARIABLE IMPORTANCE
############################

message("\nAnalyzing feature importance...")

# Select best model
best_model <- switch(
  best_model_name,
  "Baseline" = base_fit,
  "Weighted" = weight_fit,
  "SMOTE" = smote_fit,
  "Downsampled" = down_fit
)

# Extract importance
importance <- as.data.frame(varImp(best_model))
importance$Feature <- rownames(importance)

# Plot
feature_importance <- ggplot(importance, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Key Drivers of Customer Churn", x = "Features", y = "Importance Score") +
  theme_minimal()

ggsave("plots/feature_importance.png", feature_importance)

# Significant Values Check
summary(best_model)


