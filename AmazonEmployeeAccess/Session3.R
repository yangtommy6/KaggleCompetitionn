train_path <- "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv"
test_path <- "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv"
library(tidymodels)
library(themis) # for dealing with imbalanced classes
library(vroom) # for fast data reading
library(randomForest) # for random forest model
library(caret) # for additional modeling utilities


# Convert ACTION to a factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Define the recipe for preprocessing
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_smote(ACTION) %>% # SMOTE for balancing the classes
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05) %>% # Collapse infrequent levels
  step_dummy(all_nominal(), -all_outcomes()) %>% # Dummy coding for categorical variables
  step_normalize(all_numeric(), -all_outcomes()) # Normalize numeric variables

# Define a model specification for the random forest
rf_spec <- rand_forest(trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

# Define the tuning grid
tune_grid <- grid_latin_hypercube(
  mtry(range = c(2, sqrt(ncol(amazonTrain) - 1))),
  min_n(range = c(1, 10)),
  size = 20
)

# Define the cross-validation procedure
cv_folds <- vfold_cv(amazonTrain, v = 5, repeats = 5)

# Perform the tuning
tuned_results <- tune_grid(
  rf_spec,
  az_recipe,
  resamples = cv_folds,
  grid = tune_grid,
  metrics = metric_set(roc_auc)
)

# Extract the best hyperparameters
best_params <- select_best(tuned_results, "roc_auc")

# Finalize the model with the best parameters
final_rf_spec <- finalize_model(rf_spec, best_params)

# Fit the final model on the training data
final_rf_fit <- fit(final_rf_spec, az_recipe, data = amazonTrain)

# Make predictions on the test set
amazonTest_preds <- predict(final_rf_fit, amazonTest, type = "prob")

# Prepare the submission file
Id <- amazonTest$id
Action <- amazonTest_preds$.pred_yes

submission <- data.frame(Id, Action)

# Save the submission file
write.csv(submission, "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/sec3.csv", row.names = FALSE)
