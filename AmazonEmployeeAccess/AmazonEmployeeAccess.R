library(tidymodels)
library(vroom)

# Load Data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")

# Ensure 'ACTION' is a Factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Recipe for Pre-processing
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

# Model Specification
my_mod <- logistic_reg() %>%
  set_engine("glm")

# Workflow
amazon_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(my_mod)

# Split Data for Validation
set.seed(123)
data_split <- initial_split(amazonTrain, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

# Fit and Evaluate Model
fit_workflow <- fit(amazon_workflow, data = train_data)

# Predict Probabilities for Validation
predictions <- predict(fit_workflow, test_data, type = "prob") %>%
  bind_cols(test_data)

# Evaluate Metrics
metrics <- metric_set(roc_auc, accuracy, precision, recall)
eval_results <- metrics(predictions, truth = ACTION, estimate = .pred_1)
print(eval_results)

# Predict Probabilities for Test Data
amazon_predictions <- predict(fit_workflow, new_data = amazonTest, type = "prob")

# Create Submission Data Frame

amazon_predictions <- predict(fit_workflow, new_data = amazonTest, type = "class")

submission_df <- data.frame(Id = seq_len(nrow(amazonTest)), Action = amazon_predictions$.pred_class)
vroom_write(x = submission_df, file = "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/submission2.csv", delim = ",")



###


library(tidymodels)
library(vroom)
library(randomForest)

# Load Data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")


# Ensure 'ACTION' is a Factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Recipe for Pre-processing
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())
# Random Forest Model Specification
rf_mod <- rand_forest() %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

# Workflow
rf_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(rf_mod)

# Define a simple grid for tuning
grid <- grid_regular(
  mtry(range = c(2, 10)),
  trees(range = c(50, 200)),
  levels = 2
)

# Tune the model
tune_results <- tune_grid(
  rf_workflow,
  resamples = bootstraps(train_data, times = 5),
  grid = grid
)

# Extract best parameters
best_params <- tune_results %>%
  select_best(metric = "accuracy")

# Update the workflow with the best parameters
final_rf_workflow <- rf_workflow %>%
  finalize_workflow(best_params)

# Fit the final model
final_fit <- fit(final_rf_workflow, data = train_data)

# Predict on the test set
amazon_predictions <- predict(final_fit, new_data = amazonTest, type = "class")

# Create the submission dataframe
submission_df <- data.frame(Id = seq_len(nrow(amazonTest)), Action = amazon_predictions$.pred_class)

# Save the predictions to a CSV file
vroom_write(x = submission_df, file = "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/rf_submission.csv", delim = ",")


#########1018 naive bayes

library(tidymodels)
library(vroom)
library(discrim)


amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
# Ensure 'ACTION' is a Factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Recipe for Pre-processing
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

# Model Specification
my_mod <- naive_Bayes() %>%
  set_engine("naivebayes")

# Workflow
amazon_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(my_mod)

# Split Data for Validation
set.seed(123)
data_split <- initial_split(amazonTrain, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

# Fit and Evaluate Model
fit_workflow <- fit(amazon_workflow, data = train_data)

# Predict Probabilities for Validation
predictions <- predict(fit_workflow, test_data, type = "prob") %>%
  bind_cols(test_data)

# Evaluate Metrics
metrics <- metric_set(roc_auc, accuracy, precision, recall)
eval_results <- metrics(predictions, truth = ACTION, estimate = .pred_1)
print(eval_results)

# Predict Probabilities for Test Data
amazon_predictions <- predict(fit_workflow, new_data = amazonTest, type = "class")

# Create Submission Data Frame
submission_df <- data.frame(Id = seq_len(nrow(amazonTest)), Action = amazon_predictions$.pred_class)

vroom_write(x = submission_df, file = "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/naivebayes1.csv", delim = ",")




##1025


# Load necessary libraries

library(kernlab)
library(pROC)

# Read the datasets
train_data <- read.csv("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
test_data <- read.csv("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")

# Splitting data into features and target variable
features <- train_data[, -1] # Excluding the ACTION column
target <- train_data$ACTION

# Train SVM classifier using kernlab
svm_model <- ksvm(as.matrix(features), as.factor(target), type = "C-svc", kernel = "vanilladot", prob.model = TRUE)

# Predict probabilities on the test data
predicted_values <- predict(svm_model, as.matrix(test_features), type = "probabilities")[,2]

# Check the first few values to ensure they were extracted correctly
head(predicted_values)

# Compute the ROC AUC score
roc_obj <- roc(target[1:length(predicted_values)], predicted_values)
auc_val <- auc(roc_obj)
print(auc_val)

# Generate the submission file
submission <- data.frame(Id = test_data$id, Action = predicted_values)
write.csv(submission, "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/svm_submission.csv", row.names = FALSE)



######

library(kernlab)
library(pROC)
library(ggplot2)

# Read the datasets
train_data <- read.csv("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")

# Examine distribution of target variable
table(train_data$ACTION)

# Plot distribution
ggplot(train_data, aes(x = factor(ACTION))) + 
  geom_bar() +
  labs(title = "Distribution of ACTION", x = "ACTION", y = "Count")

library(ROSE)

# Oversample the minority class using SMOTE with ROSE
balanced_data <- ovun.sample(ACTION ~ ., data=train_data, method="over", N=30872*2)$data
table(balanced_data$ACTION)

library(caret)

# Setting up train control for cross validation and search grid
ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(.sigma = c(0.01, 0.05, 0.1, 0.5, 1), .C = c(1, 10, 50, 100))

# Setting up and running the train function for parameter tuning
set.seed(123) # for reproducibility
tune_model <- train(ACTION ~ ., data = balanced_data, method = "svmRadial", 
                    trControl = ctrl, tuneGrid = grid, preProcess = c("center", "scale"))

# Display the best parameters
#print(tune_model$bestTune)


# Re-training SVM with best parameters
svm_model <- ksvm(as.matrix(balanced_data[, -1]), as.factor(balanced_data$ACTION), type = "C-svc", 
                  kernel = "rbfdot", C = tune_model$bestTune$.C, kpar=list(sigma=tune_model$bestTune$.sigma))

# Predict on test set
test_features <- test_data[, -1] # Excluding the id column
predicted_values <- predict(svm_model, as.matrix(test_features), type = "probabilities")[,2]

# Generate the submission file
submission <- data.frame(Id = test_data$id, Action = predicted_values)
write.csv(submission, "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/svm_balanced_submission.csv", row.names = FALSE)



#####Oct30

library(tidymodels)
library(themis)
library(vroom)
library(randomForest)
library(xgboost)

# Read in the data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")

# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Create a recipe with SMOTE
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_smote(ACTION, neighbors = 5) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(~ ROLE_TITLE:ROLE_DEPTNAME)  # Interaction term

# Define the Random Forest model
rf_model <- rand_forest(mtry = tune(), trees = 1000, min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

# Create a workflow with the recipe and Random Forest model
rf_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values for Random Forest (expanded grid)
tuning_grid_rf <- expand.grid(
  mtry = c(2, 3, 4, 5, 6, 7, 8),
  min_n = c(2, 5, 10, 15, 20)
)

# Stratified K-fold CV
folds <- vfold_cv(amazonTrain, v = 10, repeats = 1, strata = ACTION)

# Find best tuning parameters for Random Forest
CV_results_rf <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid_rf,
    metrics = metric_set(roc_auc)
  )

# Find the best tuning parameters based on ROC AUC for Random Forest
bestTune_rf <- CV_results_rf %>%
  select_best("roc_auc")

# Finalize the workflow and fit the Random Forest model
final_wf_rf <- rf_workflow %>%
  finalize_workflow(bestTune_rf) %>%
  fit(data = amazonTrain)

# Predict on new data using the Random Forest model
amazon_predictions_rf <- predict(final_wf_rf, new_data = amazonTest, type = "prob")

# Create an ID column
Id <- 1:nrow(amazonTest)
Action_rf <- amazon_predictions_rf$.pred_1

submission_df_rf <- data.frame(Id = Id, Action = Action_rf)

# Write the submission data frame to a CSV file
vroom_write(x = submission_df_rf, file = "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/rf.csv", delim = ",")
###############################################################
library(tidymodels)
library(themis)
library(vroom)
library(xgboost)

# Read in the data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")

# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Frequency encoding for RESOURCE
resource_freq <- amazonTrain %>% count(RESOURCE) %>% rename(RESOURCE_freq = n)
amazonTrain <- left_join(amazonTrain, resource_freq, by = "RESOURCE")
amazonTest <- left_join(amazonTest, resource_freq, by = "RESOURCE")

# Create a recipe with SMOTE and other preprocessing
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_smote(ACTION, neighbors = 5) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(~ ROLE_TITLE:ROLE_DEPTNAME)

# Define the XGBoost model
xgb_model <- boost_tree(
  mode = "classification",
  trees = 1000,
  mtry = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Create a workflow
xgb_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(xgb_model)

# Stratified K-fold CV
folds <- vfold_cv(amazonTrain, v = 10, repeats = 1, strata = ACTION)

# Find best tuning parameters for XGBoost
CV_results_xgb <- xgb_workflow %>%
  tune_grid(
    resamples = folds,
    grid = 20,  # Use a larger grid or a predefined grid
    metrics = metric_set(roc_auc)
  )

# Find the best tuning parameters based on ROC AUC for XGBoost
bestTune_xgb <- CV_results_xgb %>%
  select_best("roc_auc")

# Finalize the workflow and fit the XGBoost model
final_wf_xgb <- xgb_workflow %>%
  finalize_workflow(bestTune_xgb) %>%
  fit(data = amazonTrain)

# Predict on new data using the XGBoost model
amazon_predictions_xgb <- predict(final_wf_xgb, new_data = amazonTest, type = "prob")

# Create an ID column
Id <- 1:nrow(amazonTest)
Action_xgb <- amazon_predictions_xgb$.pred_1

submission_df_xgb <- data.frame(Id = Id, Action = Action_xgb)

# Write the submission data frame to a CSV file
vroom_write(x = submission_df_xgb, file = "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/amazon_xgb.csv", delim = ",")
###############################another try, last one got a 0.86 score
library(tidymodels)
library(themis) # for SMOTE
library(vroom)
library(randomForest)
library(caret) # for additional model metrics and resampling

# Read in the data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")


# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Create a recipe
# Note: You may want to explore feature engineering and interactions.
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_smote(ACTION, neighbors = 5) %>% # Adjust neighbors based on over/under sampling
  step_other(all_nominal_predictors(), threshold = 0.05) %>% # Adjust the threshold
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% # One-hot encoding
  step_normalize(all_numeric_predictors()) # Normalize numeric predictors

# Define the Random Forest model with a more extensive tuning grid
rf_model <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

# Create a workflow
rf_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(rf_model)

# Set up a more comprehensive tuning grid
tuning_grid_rf <- expand.grid(
  mtry = seq(from = 2, to = sqrt(ncol(amazonTrain)), by = 1), # Adjust based on feature count
  min_n = c(1, 2, 5, 10),
  trees = c(500, 750, 1000, 1250) # Adjust number of trees
)

# Set up repeated K-fold cross-validation for a more robust validation
folds <- vfold_cv(amazonTrain, v = 5, repeats = 5)

# Tune the model using AUC as the metric
tune_results_rf <- tune_grid(
  rf_workflow,
  resamples = folds,
  grid = tuning_grid_rf,
  metrics = metric_set(roc_auc)
)

# Find the best hyperparameters based on ROC AUC
best_params_rf <- tune_results_rf %>%
  select_best("roc_auc")

# Update the workflow with the best parameters
final_rf_workflow <- rf_workflow %>%
  finalize_workflow(best_params_rf)

# Fit the final model on the training data
final_rf_fit <- fit(final_rf_workflow, data = amazonTrain)

# Predict on the test set
amazon_predictions_rf <- predict(final_rf_fit, new_data = amazonTest, type = "prob")

# Prepare the submission file
amazonTest$ACTION <- amazon_predictions_rf$.pred_1
submission <- amazonTest %>% 
  select(Id = row_number(), ACTION)

# Write the submission to a CSV file
write_csv(submission, "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/1.csv")
