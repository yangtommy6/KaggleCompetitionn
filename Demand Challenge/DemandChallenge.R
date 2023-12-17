# Load necessary libraries
library(ggplot2)
library(forecast)

# Read the data
test_data <- read.csv("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/test.csv")
train_data <- read.csv("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv")
library(forecast)
view(train_data)
# Function to generate and save ACF plot
generate_acf_plot <- function(data, store, item, filepath) {
  data_filtered <- data[data$store == store & data$item == item, ]
  png(filepath, width=800, height=400)
  Acf(data_filtered$sales, main=paste("ACF Plot - Store", store, "Item", item))
  dev.off()
}

# Read the data
data <- read.csv('/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv')

# Define the combinations
combinations <- data.frame(store=c(5, 1, 8, 8), item=c(31, 35, 5, 7))

# Generate and save ACF plots for each combination
for (i in 1:nrow(combinations)) {
  store <- combinations$store[i]
  item <- combinations$item[i]
  filepath <- paste('/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/acf_plot_store', store, '_item', item, '.png', sep='')
  generate_acf_plot(data, store, item, filepath)
}


###
library(forecast)

# Function to generate ACF plot
generate_acf_plot <- function(data, store, item) {
  data_filtered <- data[data$store == store & data$item == item, ]
  Acf(data_filtered$sales, main=paste("Store", store, "Item", item))
}

# Read the data
data <- read.csv('/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv')

# Define the combinations
combinations <- data.frame(store=c(5, 1, 8, 8), item=c(31, 35, 5, 7))

# Setup the layout for 4 plots and create a PNG file
png('/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/combined_acf_plot.png', width=800, height=800)
par(mfrow=c(2, 2))

# Generate ACF plots for each combination
for (i in 1:nrow(combinations)) {
  store <- combinations$store[i]
  item <- combinations$item[i]
  generate_acf_plot(data, store, item)
}

# Close the PNG device
dev.off()




###########library(tidymodels)
library(dplyr)
library(lubridate)
library(forecast)
library(yardstick)
library(vroom)

item_Train <- vroom("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv") 
item_Test <- vroom("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/test.csv") 

storeItem <- item_Train %>%
  filter(store==2, item==7)

# Define your model (using a random forest model as an example)
model_spec <- rand_forest(trees = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")

# Adjusting the recipe
recipe <- recipe(sales ~ ., data = storeItem) %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy)) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  step_normalize(all_numeric_predictors(), -all_outcomes())

# Define resampling strategy (using time series cross-validation)
time_series_cv <- rolling_origin(storeItem, initial = 365 * 2, assess = 365, skip = 365)

# Define the metric set to include sMAPE
metric_set <- metric_set(smape)

# Tuning the hyperparameters
cv_results <- tune_grid(
  model_spec,
  recipe,
  resamples = time_series_cv,
  grid = 10, # or use a predefined grid
  metrics = metric_set(smape)  # Using smape directly
)

# Collect the metrics from the tuning results
collected_metrics <- collect_metrics(cv_results)


# Filter for the best sMAPE metric
# Filter for the best sMAPE metric using top_n
best_smape <- collected_metrics %>%
  filter(.metric == "smape") %>%
  arrange(mean) %>%
  top_n(-1, mean) %>%
  pull(mean)


# The 'best_smape' variable now holds the mean sMAPE value of the best model
print(best_smape)



####1027


# Load required libraries
library(readr)
library(dplyr)
library(tsibble)

# Load the training data
train_data <- read_csv("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv") # Replace with the correct file path
# Load required libraries
library(readr)
library(dplyr)
library(tsibble)
library(fable)
library(recipes)
library(lubridate) # For date manipulation


# Filter data for the selected combination (Store 1, Item 10 as an example)
combo1_data <- train_data %>% 
  filter(store == 1, item == 10) %>% 
  select(date, sales) %>% 
  mutate(date = as.Date(date, format = "%Y-%m-%d"))

# Convert to tsibble (time series tibble)
combo1_ts <- combo1_data %>% as_tsibble(index = date)

# Define the training and validation periods
training_period_end <- as.Date("2017-12-31") # Adjust as needed
validation_period_start <- as.Date("2018-01-01") # Adjust as needed
validation_period_end <- as.Date("2018-03-31") # Adjust as needed

# Split the data into training and validation sets, ensuring they are tsibble objects
training_data <- combo1_ts %>% filter(date <= training_period_end)
validation_data <- combo1_ts %>% filter(date >= validation_period_start & date <= validation_period_end)

# Fit ARIMA model on the training data
fit_combo1 <- training_data %>%
  model(ARIMA(sales ~ pdq() + PDQ()))

# ...
forecast_horizon <- as.numeric(difftime(validation_period_end, validation_period_start, units = "days")) + 1
forecast_combo1 <- forecast(fit_combo1, h = forecast_horizon)

# To ensure alignment, we'll create a new data frame that combines the forecast with the validation dates
forecast_df <- data.frame(date = seq.Date(validation_period_start, validation_period_end, by = "day"),
                          forecast = as.numeric(forecast_combo1$.mean))

# ...

# ...

# Inspect the date range and format in validation_data
print(min(validation_data$date))
print(max(validation_data$date))

# Inspect the date range and format in forecast_df
print(min(forecast_df$date))
print(max(forecast_df$date))

# Join with the validation data
validation_combined <- left_join(validation_data, forecast_df, by = "date")

# Inspect the combined data again
print(head(validation_combined))

# If the join is still not successful, check for issues like different date formats or missing dates

# Assuming the join is successful now
# Calculate accuracy metrics
accuracy_metrics <- accuracy(validation_combined$forecast, validation_combined$sales)

# Print accuracy metrics
print(accuracy_metrics)

# ...




# Forecast for future periods (e.g., next 3 months)
future_forecast <- forecast(fit_combo1, h = "3 months")

# Plotting the forecast
plot(future_forecast, main = "Forecast for Next 3 Months")



# libraries
# install.packages('lubridate')
library(lubridate)

library(astsa)
library(forecast)
options(warn=-1) # ignore warnings globally
# read data to R variable
demand.data <- read.csv("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv")

# use date format for dates
demand.data$date <- as.Date(demand.data$date, "%Y-%m-%d")

head(demand.data)
# read data to R variable
test.data <- read.csv("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/test.csv")

# use date format for dates
test.data$date <- as.Date(test.data$date, "%Y-%m-%d")

head(test.data)

# Subset Store 1 and Item 1
data.1.1 <- subset(demand.data, store == 1 & item == 1 , select=c('date','sales'))

x1 <- ts(data.1.1$sales, frequency=365.25)
fit1 <- tbats(x1)
seasonal1 <- !is.null(fit1$seasonal)
cat("Yearly seasonal : ",seasonal1)

x2 <- ts(data.1.1$sales, frequency=7)
fit2 <- tbats(x2)
seasonal2 <- !is.null(fit2$seasonal)
cat("\nWeekly seasonal : ",seasonal2)

# Create lists
item <- list()
sub <- list()

for(i in 1:50) { # items
  for(j in 1:10){ # stores
    # create separate dataframes
    assign(paste("demand.data",j,i,sep="."),subset(demand.data, store == j & item == i )) -> data
    
    # create msts
    assign(paste("msts",j,i,sep="."), msts(data$sales,seasonal.periods = c(7,365.25),start = decimal_date(as.Date("2013-01-01")))) -> msts
    
    # TBATS model
    assign(paste("model",j,i,sep="."), tbats(msts)) -> model
    
    # Forecast
    assign(paste("forecast",j,i,sep="."), forecast(model,h=90)) -> forecast
    assign(paste("forecast.sales",j,i,sep="."), round(data.frame(forecast)$Point.Forecast,0)) -> forecast.sales
    assign(paste("forecast.sales",j,i,sep="."), data.frame(forecast.sales)) -> forecast.sales    
    names(forecast.sales) <- "sales"
    
    item[[j]] <- forecast.sales   # add it to your list 
  }
  assign(paste("item",i,sep="."),do.call(rbind,item) ) -> ii
  
  sub[[i]] <- ii # add it to your list 
  
}

assign(paste("demand.sales"),do.call(rbind,sub)) 

# submission

submission <- cbind(test.data,demand.sales)
submission <- submission[,c("id","sales")]

head(submission)

write.csv(submission, file = "/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/submission.csv", row.names = F)






library(lubridate)
library(forecast)

# Function to read and preprocess data
read_and_process_data <- function(file_path) {
  data <- read.csv(file_path)
  data$date <- as.Date(data$date, "%Y-%m-%d")
  return(data)
}

# Load data
train_data <- read_and_process_data("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/train.csv")
test_data <- read_and_process_data("/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/test.csv")

# Function to model and forecast sales for a given store and item
forecast_sales <- function(store_id, item_id, training_data, h = 90) {
  # Subset data
  data_subset <- subset(training_data, store == store_id & item == item_id)
  
  # Create msts object
  msts_data <- msts(data_subset$sales, seasonal.periods = c(7, 365.25), start = decimal_date(as.Date("2013-01-01")))
  
  # Fit TBATS model
  model <- tbats(msts_data)
  
  # Forecast
  forecasted_values <- forecast(model, h = h)
  return(round(data.frame(forecasted_values)$Point.Forecast, 0))
}

# Initialize a list to store forecasts
forecasts <- list()

# Loop over each store and item to forecast sales
for (i in 1:50) { # Items
  for (j in 1:10) { # Stores
    forecasts[[paste("item", i, "store", j, sep = "_")]] <- forecast_sales(j, i, train_data)
  }
}

# Prepare submission data
submission <- cbind(test_data, unlist(forecasts, recursive = FALSE))
submission <- submission[, c("id", "sales")]

# Save submission file
write.csv(submission, file = "/Users/christian/Desktop/STAT348/DemandChallenge/Demand-Challenge/submission.csv", row.names = FALSE)

