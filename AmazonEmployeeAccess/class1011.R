
library(tidymodels)
library(embed)  # for step_encode()
library(vroom)

# Load Data
amazonTrain <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/test.csv")


# Ensure 'ACTION' is a Factor
amazonTrain$ACTION <- factor(amazonTrain$ACTION)

# Recipe for Pre-processing with Target Encoding
az_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())


# Model Specification with Penalized Logistic Regression
my_mod <- logistic_reg(penalty = tune(), mixture = tune()) %>%  
  set_engine("glmnet")

# Workflow
amazon_workflow <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(my_mod)

# Split Data for Validation
set.seed(123)
data_split <- initial_split(amazonTrain, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

# Model Tuning
set.seed(123)
grid_vals <- grid_regular(penalty(), mixture(), levels = 5)
amazon_res <- tune_grid(
  amazon_workflow,
  resamples = bootstraps(train_data, times = 5),
  grid = grid_vals,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

# Extract best parameters
best_params <- select_best(amazon_res, "roc_auc")
print(best_params)

# Fit with best parameters
final_model <- finalize_model(my_mod, best_params)
final_workflow <- amazon_workflow %>% update_model(final_model)
fit_workflow <- fit(final_workflow, data = train_data)

# Predict and Evaluate Metrics
predictions <- predict(fit_workflow, test_data, type = "prob") %>%
  bind_cols(test_data)

# Evaluate Metrics
metrics <- metric_set(roc_auc, accuracy, precision, recall)
eval_results <- metrics(predictions, truth = ACTION, estimate = .pred_1)
print(eval_results)

# Predict for Test Data
amazon_predictions <- predict(fit_workflow, new_data = amazonTest, type = "prob")

# Create Submission Data Frame
submission_df <- data.frame(Id = seq_len(nrow(amazonTest)), Action = amazon_predictions$.pred_1)

# Save submission to CSV
output_path <- "/Users/christian/Desktop/STAT348/AmazonEmployeeAccess/submission3.csv"  # Update path
vroom_write(x = submission_df, file = output_path, delim = ",")
