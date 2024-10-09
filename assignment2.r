# Install required packages if you haven't already
# install.packages("readr")
# install.packages("dplyr")
# install.packages("caTools")
# install.packages("rpart")
# install.packages("caret")
# install.packages("randomForest")
# install.packages("ggplot2")
# install.packages("class")

# Load required libraries
library(readr)
library(dplyr)
library(caTools)
library(rpart)
library(caret)
library(randomForest)
library(ggplot2)
library(class)

# Fetch the dataset from UCI
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

# Load the dataset into R
car_evaluation <- read_csv(url, col_names = FALSE)

# Assign feature and target variables (assuming the last column is the target 'CAR')
colnames(car_evaluation) <- c('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car_acceptability')

# Split the dataset into features and target
X <- car_evaluation %>% select(-car_acceptability)
y <- car_evaluation$car_acceptability

# Split the data into training and testing sets (70% train, 30% test)
set.seed(42)
split <- sample.split(y, SplitRatio = 0.7)

X_train <- X[split, ]
X_test <- X[!split, ]
y_train <- y[split]
y_test <- y[!split]

# Print training and test set sizes
cat("Training set size:", nrow(X_train), "\n")
cat("Test set size:", nrow(X_test), "\n")

# Decision Tree model
dt_model <- rpart(y_train ~ ., data = X_train, method = "class", control = rpart.control(cp = 0, minsplit = 2))

# Predictions using Decision Tree
y_pred_dt <- predict(dt_model, X_test, type = "class")

# Confusion matrix for Decision Tree
confusionMatrix(factor(y_test), factor(y_pred_dt))

# Random Forest model
rf_model <- randomForest(as.factor(y_train) ~ ., data = X_train, ntree = 100)

# Predictions using Random Forest
y_pred_rf <- predict(rf_model, X_test)

# Confusion Matrix for Random Forest
confusion_rf <- table(Predicted = y_pred_rf, Actual = as.factor(y_test))
print(confusion_rf)

# Accuracy for Random Forest
accuracy_rf <- sum(diag(confusion_rf)) / sum(confusion_rf)
cat("Random Forest Accuracy:", accuracy_rf, "\n")

# Precision, Recall, and F1-Score for Random Forest
precision_rf <- diag(confusion_rf) / rowSums(confusion_rf)
recall_rf <- diag(confusion_rf) / colSums(confusion_rf)
f1_score_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

# Combine metrics into a data frame for better viewing
metrics_rf <- data.frame(
  Class = names(precision_rf),
  Precision = precision_rf,
  Recall = recall_rf,
  F1_Score = f1_score_rf
)
print(metrics_rf)

# k-NN classification (k = 5)
y_pred_knn <- knn(train = X_train, test = X_test, cl = as.factor(y_train), k = 5)

# Confusion matrix for k-NN
confusion_knn <- table(Predicted = y_pred_knn, Actual = as.factor(y_test))
print(confusion_knn)

# Check if the confusion matrix is square and print metrics
if (nrow(confusion_knn) == ncol(confusion_knn)) {
  confusion_metrics_knn <- confusionMatrix(confusion_knn)
  print(confusion_metrics_knn)
}

# Calculate accuracy for Decision Tree
acc_dt <- sum(y_pred_dt == y_test) / length(y_test)
cat(sprintf("Decision Tree Accuracy: %.2f\n", acc_dt))

# Calculate accuracy for k-NN
acc_knn <- sum(y_pred_knn == y_test) / length(y_test)
cat(sprintf("k-NN Accuracy: %.2f\n", acc_knn))

# Feature importance for Random Forest
importances <- rf_model$importance
indices <- order(importances, decreasing = TRUE)

# Prepare data for plotting feature importance
importance_data <- data.frame(Feature = colnames(X_train)[indices], Importance = importances[indices])

# Plot the feature importance
ggplot(importance_data, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance") +
  theme_minimal()
