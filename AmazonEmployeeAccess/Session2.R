# Read in the data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
library(tidymodels)
library(themis)
library(vroom)
library(randomForest)
library(dials) # for randomized search
library(DMwR2) # for SMOTE


# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Set up a recipe
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_smote(ACTION, neighbors = 5) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Define a model spec for random forest
rf_model <- rand_forest(mode = "classification", trees = 1000) %>%
  set_engine("randomForest") %>%
  set_args(mtry = tune(), min_n = tune())

# Set up 10-fold cross-validation
set.seed(123)
folds <- vfold_cv(amazonTrain, v = 10, strata = ACTION)

# Tune the model
tuning_results <- tune_grid(
  rf_model,
  az_recipe,
  resamples = folds,
  grid = 20,
  metrics = metric_set(roc_auc)
)

# Get the best hyperparameters
best_params <- select_best(tuning_results, "roc_auc")

# Update the model spec with the best hyperparameters
final_rf_model <- finalize_model(rf_model, best_params)

# Fit the final model to the training data
final_fit <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(final_rf_model) %>%
  fit(data = amazonTrain)

# Predict on the test data
amazon_predictions <- predict(final_fit, amazonTest, type = "prob")

# Prepare the submission data frame
submission_df <- amazon_predictions %>%
  bind_cols(amazonTest %>% select(ID = 1)) %>%
  select(ID, .pred_class = .pred_1)

# Write the submission file
submission_file <- "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/2.csv"
write_csv(submission_df, submission_file)
