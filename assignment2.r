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
  head(data,20)
  cat(unique(data$buying))
  cat(unique(data$maint))
  cat(unique(data$doors))
  cat(unique(data$persons))
  cat(unique(data$lug_boot))
  cat(unique(data$safety))
  cat(unique(data$class))
  for (column in colnames(data)) {
    data[[column]] <- as.numeric(factor(data[[column]]))
  }
  return(data)
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

# Function to train and evaluate Decision Tree model
train_decision_tree <- function(X_train, y_train, X_test, y_test) {
  dt_model <- rpart(y_train ~ ., data = X_train, method = "class", control = rpart.control(cp = 0, minsplit = 2))
  y_pred <- predict(dt_model, X_test, type = "class")
  accuracy <- sum(y_pred == y_test) / length(y_test)
  cat(sprintf("Decision Tree Accuracy: %.2f\n", accuracy))
  return(list(model = dt_model, predictions = y_pred, accuracy = accuracy))
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
  return(list(model = rf_model, predictions = y_pred, accuracy = accuracy))
}

# Function to train and evaluate k-NN model
train_knn <- function(X_train, y_train, X_test, y_test, k = 5) {
  y_pred <- knn(train = X_train, test = X_test, cl = as.factor(y_train), k = k)
  accuracy <- sum(y_pred == y_test) / length(y_test)
  cat(sprintf("k-NN Accuracy: %.2f\n", accuracy))
  return(list(predictions = y_pred, accuracy = accuracy))
}

# Function to plot feature importance for Random Forest
plot_feature_importance_rf <- function(rf_model) {
  importance_rf <- as.data.frame(importance(rf_model))
  importance_rf$Feature <- rownames(importance_rf)
  ggplot(importance_rf, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(title = "Feature Importance - Random Forest", x = "Features", y = "Importance")
}

# Function to plot feature importance for Decision Tree
plot_feature_importance_dt <- function(dt_model) {
  # Extract feature importance from the decision tree model
  importance_dt <- as.data.frame(dt_model$variable.importance)
  importance_dt$Feature <- rownames(importance_dt)
  colnames(importance_dt)[1] <- "Importance"  # Rename column
  
  ggplot(importance_dt, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() +
    labs(title = "Feature Importance - Decision Tree", x = "Features", y = "Importance")
}

# Function to plot ROC curves for models
plot_multiclass_roc_curve <- function(y_test, y_pred, model_name) {
  # Convert to numeric for multiclass ROC
  y_test_numeric <- as.numeric(as.factor(y_test))
  y_pred_numeric <- as.numeric(as.factor(y_pred))
  
  roc_obj <- multiclass.roc(y_test_numeric, y_pred_numeric)
  
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate",
       main = paste(model_name, "ROC Curve"))
  colors <- rainbow(length(roc_obj$rocs))  
  
  for (i in 1:length(roc_obj$rocs)) {
    plot.roc(roc_obj$rocs[[i]], col = colors[i], add = TRUE)
  }
  
  legend("bottomleft", legend = paste("Class", unique(y_test)), col = colors, lwd = 2)
}

plot_roc_curve <- function(y_test, y_pred, model_name) {
  y_test_numeric <- as.numeric(as.factor(y_test))
  y_pred_numeric <- as.numeric(as.factor(y_pred))
  roc_obj <- multiclass.roc(y_test_numeric, y_pred_numeric)
  plot.roc(roc_obj$rocs[[1]], col = "blue", main = paste(model_name, "ROC Curve"))
}


# Function to load and preprocess the dataset
load_and_preprocess_data <- function(url, columns) {
  data <- read_csv(url, col_names = columns)
  for (column in colnames(data)) {
    data[[column]] <- as.numeric(factor(data[[column]]))
  }
  return(as.data.frame(data))
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

# Feature Importance for Random Forest
plot_feature_importance_rf(rf_results$model)
plot_feature_importance_dt(dt_results$model)

# ROC Curves
plot_multiclass_roc_curve(y_test, dt_results$predictions, "Decision Tree")
plot_multiclass_roc_curve(y_test, rf_results$predictions, "Random Forest")

plot_roc_curve(y_test, rf_results$predictions, "Random Forest")
plot_roc_curve(y_test, dt_results$predictions, "Decision Tree")
