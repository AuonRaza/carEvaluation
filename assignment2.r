# Function to install necessary packages
install_required_packages <- function() {
  install.packages("readr")
  install.packages("dplyr")
  # install.packages("caret")
  install.packages("randomForest")
  # install.packages(c("ggplot2", "lattice", "data.table", "dplyr", "Matrix", "pROC", "tidyverse"))
  install.packages("curl")
  install.packages("gargle")
  # install.packages("googledrive")
  install.packages("httr")
  install.packages("ragg")
  install.packages("rvest")
  # install.packages("ggplot2")
  # install.packages("googlesheets4")
  install.packages("tidyverse")
  # install.packages("caret")
  install.packages("scales")
  install.packages("ggplot2")
  install.packages("caret")
}

# Function to load required libraries
load_libraries <- function() {
  library(readr)
  library(dplyr)
  library(caTools)
  library(rpart)
  library(caret)
  library(ggplot2)
  library(randomForest)
  library(class) # for k-NN
  library(pROC)
}

# Function to load and preprocess the dataset
load_and_preprocess_data <- function(url, columns) {
  data <- read_csv(url, col_names = columns)
  for (column in colnames(data)) {
    data[[column]] <- as.numeric(factor(data[[column]]))
  }
  return(data)
}

# Function to split data into training and testing sets
split_data <- function(data, target_column, split_ratio = 0.7) {
  X <- data[, !names(data) %in% target_column]
  y <- data[[target_column]]
  set.seed(42)
  split <- sample.split(y, SplitRatio = split_ratio)
  X_train <- X[split, ]
  X_test <- X[!split, ]
  y_train <- y[split]
  y_test <- y[!split]
  return(list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test))
}

# Function to train and evaluate Decision Tree model
train_decision_tree <- function(X_train, y_train, X_test, y_test) {
  dt_model <- rpart(y_train ~ ., data = X_train, method = "class", control = rpart.control(cp = 0, minsplit = 2))
  y_pred <- predict(dt_model, X_test, type = "class")
  accuracy <- sum(y_pred == y_test) / length(y_test)
  cat(sprintf("Decision Tree Accuracy: %.2f\n", accuracy))
  return(list(model = dt_model, accuracy = accuracy))
}

# Function to train and evaluate Random Forest model


train_random_forest <- function(X_train, y_train, X_test, y_test) {
  rf_model <- randomForest(as.factor(y_train) ~ ., data = X_train, ntree = 100)
  y_pred <- predict(rf_model, X_test)

  # Ensure the factor levels of y_pred and y_test are consistent
  all_classes <- union(levels(as.factor(y_train)), levels(as.factor(y_test)))
  y_pred <- factor(y_pred, levels = all_classes)
  y_test <- factor(y_test, levels = all_classes)

  confusion <- table(Predicted = y_pred, Actual = y_test)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  cat("Random Forest Accuracy:", accuracy, "\n")
  return(list(model = rf_model, accuracy = accuracy))
}

# Function to train and evaluate k-NN model
train_knn <- function(X_train, y_train, X_test, y_test, k = 5) {
  y_pred <- knn(train = X_train, test = X_test, cl = as.factor(y_train), k = k)
  accuracy <- sum(y_pred == y_test) / length(y_test)
  cat(sprintf("k-NN Accuracy: %.2f\n", accuracy))
  return(list(predictions = y_pred, accuracy = accuracy))
}

# Function to plot feature importance for Random Forest
plot_feature_importance <- function(rf_model) {
  importance_rf <- as.data.frame(importance(rf_model))
  importance_rf$Feature <- rownames(importance_rf)
  ggplot(importance_rf, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(title = "Feature Importance - Random Forest", x = "Features", y = "Importance")
}

# Function to plot ROC curves for models
plot_roc_curve <- function(y_test, y_pred, model_name) {
  roc_obj <- multiclass.roc(as.numeric(y_test), as.numeric(y_pred))
  plot.roc(roc_obj$rocs[[1]], col = "blue", main = paste(model_name, "ROC Curve"))
}

plot_roc_curve <- function(y_test, y_pred, model_name) {
  y_test_numeric <- as.numeric(as.factor(y_test))
  y_pred_numeric <- as.numeric(as.factor(y_pred))
  roc_obj <- multiclass.roc(y_test_numeric, y_pred_numeric)
  plot.roc(roc_obj$rocs[[1]], col = "blue", main = paste(model_name, "ROC Curve"))
}

# Main Execution
install_required_packages()
load_libraries()

# Function to load and preprocess the dataset
load_and_preprocess_data <- function(url, columns) {
  data <- read_csv(url, col_names = columns)
  for (column in colnames(data)) {
    data[[column]] <- as.numeric(factor(data[[column]]))
  }
  return(as.data.frame(data))
}

# Function to split data into training and testing sets
split_data <- function(data, target_column, split_ratio = 0.7) {
  X <- data[, !names(data) %in% target_column]
  y <- as.vector(unlist(data[[target_column]])) # Ensure y is a vector
  set.seed(42)
  split <- sample.split(y, SplitRatio = split_ratio)
  X_train <- X[split, ]
  X_test <- X[!split, ]
  y_train <- y[split]
  y_test <- y[!split]
  return(list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test))
}

# Function to train and evaluate k-NN model
train_knn <- function(X_train, y_train, X_test, y_test, k = 5) {
  X_train <- as.data.frame(X_train)
  X_test <- as.data.frame(X_test)
  y_train <- as.factor(y_train)
  y_test <- as.factor(y_test)

  y_pred <- knn(train = X_train, test = X_test, cl = y_train, k = k)
  accuracy <- sum(y_pred == y_test) / length(y_test)
  cat(sprintf("k-NN Accuracy: %.2f\n", accuracy))
  return(list(predictions = y_pred, accuracy = accuracy))
}

# Main Execution - Ensure Proper Installation and Loading of Packages
install_required_packages()
load_libraries()

# URLs and columns
url <- "https://archive.ics.uci.edu/static/public/19/data.csv"
columns <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")

# Load and preprocess data
data <- load_and_preprocess_data(url, columns)
data <- data[-1, ]

# Split data
splits <- split_data(data, target_column = "class")
X_train <- splits$X_train
X_test <- splits$X_test
y_train <- splits$y_train
y_test <- splits$y_test

# Train models
dt_results <- train_decision_tree(X_train, y_train, X_test, y_test)
rf_results <- train_random_forest(X_train, y_train, X_test, y_test)
knn_results <- train_knn(X_train, y_train, X_test, y_test, k = 5)

