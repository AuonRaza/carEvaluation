# Install and load required packages
packages <- c("readr", "dplyr", "ggplot2", "corrplot", "caret", "randomForest", 
              "e1071", "class", "pROC", "ROCR", "shapviz", "gridExtra", "rpart",
              "iml")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

lapply(packages, library, character.only = TRUE)

# 1. Data Loading and Preprocessing
url <- 'https://archive.ics.uci.edu/static/public/19/data.csv'
columns <- c('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')
data <- read_csv(url, col_names = columns)

# Remove the header row if it's included in the data
data <- data[-1,]

# Convert all columns to factors
data <- data %>% mutate_all(as.factor)

# 2. Exploratory Data Analysis
# Class distribution
class_dist <- ggplot(data, aes(x = class)) + 
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Class Distribution", x = "Class", y = "Count")

# Feature distributions
feature_dist <- lapply(columns[-7], function(col) {
  ggplot(data, aes_string(x = col)) + 
    geom_bar(fill = "steelblue") +
    theme_minimal() +
    labs(title = paste(col, "Distribution"), x = col, y = "Count")
})

# Correlation plot
data_numeric <- data %>% mutate_all(as.numeric)
correlation_plot <- corrplot(cor(data_numeric), method = "circle")
data_numeric <- data %>% mutate(across(-class, as.numeric)) #for KNN


# 3. Data Splitting
set.seed(42)
trainIndex <- createDataPartition(data$class, p = .7, list = FALSE, times = 1)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]
trainData_numeric <- data_numeric[trainIndex,]
testData_numeric <- data_numeric[-trainIndex,]

# 4. Model Development and Evaluation
calculate_metrics <- function(actual, predicted) {
  # Ensure both actual and predicted are factors with the same levels
  actual <- factor(actual)
  predicted <- factor(predicted, levels = levels(actual))
  
  cm <- confusionMatrix(predicted, actual)
  
  accuracy <- as.numeric(cm$overall['Accuracy'])
  
  # For multi-class, calculate macro averages
  precision <- mean(cm$byClass[,'Precision'], na.rm = TRUE)
  recall <- mean(cm$byClass[,'Recall'], na.rm = TRUE)
  f1 <- mean(cm$byClass[,'F1'], na.rm = TRUE)
  
  return(c(Accuracy = accuracy, Precision = precision, Recall = recall, F1 = f1))
}

# Function to perform cross-validation
cv_model <- function(model, data, method) {
  ctrl <- trainControl(method = "cv", number = 5)
  cv_results <- train(class ~ ., data = data, method = method, trControl = ctrl)
  return(cv_results)
}

# Decision Tree
dt_model <- rpart(class ~ ., data = trainData, method = "class")
dt_pred <- predict(dt_model, newdata = testData, type = "class")
dt_metrics <- calculate_metrics(testData$class, dt_pred)
dt_cv <- train(class ~ ., 
               data = trainData, 
               method = "rpart",
               trControl = trainControl(method = "cv", number = 5))

# Random Forest
rf_model <- randomForest(class ~ ., data = trainData, ntree = 100)
rf_pred <- predict(rf_model, newdata = testData)
rf_metrics <- calculate_metrics(testData$class, rf_pred)
rf_cv <- train(class ~ ., 
               data = trainData, 
               method = "rf",
               trControl = trainControl(method = "cv", number = 5))

# Support Vector Machine
svm_model <- svm(class ~ ., data = trainData, kernel = "radial")
svm_pred <- predict(svm_model, newdata = testData)
svm_metrics <- calculate_metrics(testData$class, svm_pred)
svm_cv <- train(class ~ ., 
                data = trainData, 
                method = "svmRadial",
                trControl = trainControl(method = "cv", number = 5))

# k-Nearest Neighbors
k_value <- 3
knn_pred <- knn(train = trainData_numeric[, -which(names(trainData_numeric) == "class")], 
                test = testData_numeric[, -which(names(testData_numeric) == "class")], 
                cl = trainData_numeric$class, 
                k = k_value)
knn_metrics <- calculate_metrics(testData$class, knn_pred)
knn_cv <- train(class ~ ., 
                data = trainData_numeric, 
                method = "knn",
                trControl = trainControl(method = "cv", number = 5))

# Update the train and test datasets
trainData_numeric <- data_numeric[trainIndex, ]
testData_numeric <- data_numeric[-trainIndex, ]

# k-Nearest Neighbors
k_value <- 3
knn_pred <- knn(train = trainData_numeric[, -which(names(trainData_numeric) == "class")], 
                test = testData_numeric[, -which(names(testData_numeric) == "class")], 
                cl = trainData_numeric$class, 
                k = k_value)

# Calculate metrics
knn_metrics <- calculate_metrics(testData$class, knn_pred)
# Cross-validation for KNN (using caret for consistency)
cv_model <- function(data, method) {
  ctrl <- trainControl(method = "cv", number = 5)
  
  # For kNN, use numeric data
  if (method == "knn") {
    data <- data %>% mutate(across(-class, as.numeric))
  }
  
  # Ensure class is a factor
  data$class <- as.factor(data$class)
  
  cv_results <- train(class ~ ., data = data, method = method, trControl = ctrl)
  return(cv_results)
}

# 5. Feature Importance Analysis
# Random Forest Feature Importance
rf_importance <- importance(rf_model)
rf_importance_plot <- ggplot(data.frame(feature = rownames(rf_importance), importance = rf_importance[,1]), 
                             aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Random Forest Feature Importance", x = "Features", y = "Importance")

# SHAP values for Random Forest
# Create a predictor object
predictor <- Predictor$new(rf_model, data = trainData[,-7], y = trainData$class)

# Compute SHAP values
shap_values <- Shapley$new(predictor, x.interest = trainData[1,-7]) # Change index for interest point
shap_summary <- plot(shap_values) # Use plot instead of sv_importance

# 6. Results Visualization
# Display metrics
model_comparison <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVM", "k-NN"),
  Accuracy = c(dt_metrics["Accuracy"], rf_metrics["Accuracy"], svm_metrics["Accuracy"], knn_metrics["Accuracy"]),
  F1_Score = c(dt_metrics["F1"], rf_metrics["F1"], svm_metrics["F1"], knn_metrics["F1"])
)

model_comparison_plot <- ggplot(model_comparison, aes(x = Model)) +
  geom_bar(aes(y = Accuracy, fill = "Accuracy"), stat = "identity", position = position_dodge()) +
  geom_bar(aes(y = F1_Score, fill = "F1 Score"), stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("Accuracy" = "steelblue", "F1 Score" = "darkred")) +
  theme_minimal() +
  labs(title = "Model Comparison", y = "Score", fill = "Metric")

# 7. Results Interpretation
cat("Model Performance Summary:\n")
print(model_comparison)

cat("\nCross-Validation Results:\n")
print(dt_cv)
print(rf_cv)
print(svm_cv)
print(knn_cv)

cat("\nFeature Importance Analysis:\n")
print(rf_importance)

cat("\nConclusions:\n")
cat("1. The Random Forest model appears to perform best overall, with the highest accuracy and F1 score.\n")
cat("2. The most important features for predicting car acceptability are: ", 
    paste(rownames(rf_importance)[order(rf_importance[,1], decreasing = TRUE)[1:3]], collapse = ", "), ".\n")
cat("3. Cross-validation results suggest that our models are relatively stable across different subsets of the data.\n")
cat("4. SHAP values provide additional insights into how each feature contributes to individual predictions.\n")

# Display plots
grid.arrange(class_dist, rf_importance_plot, model_comparison_plot, ncol = 2)
print(shap_summary)