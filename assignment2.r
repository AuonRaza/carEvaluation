# Install required packages if you haven't already
install.packages("readr", "/home/shahzain/R-libraries")
install.packages("dplyr","/home/shahzain/R-libraries")
install.packages("caret", "/home/shahzain/R-libraries")
install.packages("randomForest", "/home/shahzain/R-libraries")
install.packages(c("ggplot2", "lattice", "data.table", "dplyr", "Matrix", "pROC", "tidyverse"), lib = "/home/shahzain/R-libraries")

# Load required libraries
library(readr)
library(dplyr)
library(caTools, lib.loc = "/home/shahzain/R-libraries")
library(rpart, lib.loc = "/home/shahzain/R-libraries")
library(caret, lib.loc = "/home/shahzain/R-libraries")
library(ggplot2, lib.loc = "/home/shahzain/R-libraries")
library(randomForest, lib.loc = "/home/shahzain/R-libraries")
library(class)  # for k-NN
library(pROC, lib.loc = "/home/shahzain/R-libraries")

# Load and preprocess the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
car_evaluation <- read_csv(url, col_names = FALSE)

# Rename columns
colnames(car_evaluation) <- c('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car_acceptability')
X <- car_evaluation %>% select(-car_acceptability)  # Features
y <- car_evaluation$car_acceptability               # Target variable

# Another dataset
url2 <- 'https://archive.ics.uci.edu/static/public/19/data.csv'
columns <- c('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')
data <- read_csv(url2, col_names = columns)
str(data)

# Encode categorical columns using factor()
for (column in colnames(data)) {
  data[[column]] <- as.numeric(factor(data[[column]]))
}

# Split data into training and testing sets
X <- data[, !names(data) %in% 'class']
y <- data$class
set.seed(42)
split <- sample.split(y, SplitRatio = 0.7)
X_train <- X[split, ]
X_test <- X[!split, ]
y_train <- y[split]
y_test <- y[!split]

# Decision Tree Model
dt_model <- rpart(y_train ~ ., data = X_train, method = "class", control = rpart.control(cp = 0, minsplit = 2))
y_pred_dt <- predict(dt_model, X_test, type = "class")
confusionMatrix(factor(y_test), factor(y_pred_dt))

# Random Forest Model
rf_model <- randomForest(as.factor(y_train) ~ ., data = X_train, ntree = 100)
y_pred_rf <- predict(rf_model, X_test)
all_classes <- union(levels(as.factor(y_train)), levels(as.factor(y_test)))
y_pred_rf <- factor(y_pred_rf, levels = all_classes)
y_test <- factor(y_test, levels = all_classes)
confusion_rf <- table(Predicted = y_pred_rf, Actual = y_test)
accuracy <- sum(diag(confusion_rf)) / sum(confusion_rf)
cat("Random Forest Accuracy:", accuracy, "\n")

# k-Nearest Neighbors (k-NN)
y_pred_knn <- knn(train = X_train, test = X_test, cl = as.factor(y_train), k = 5)
confusion_knn <- table(Predicted = y_pred_knn, Actual = as.factor(y_test))
if (nrow(confusion_knn) == ncol(confusion_knn)) {
  confusion_metrics <- confusionMatrix(confusion_knn)
  print(confusion_metrics)
}

# Accuracy Calculation for All Models
acc_dt <- sum(y_pred_dt == y_test) / length(y_test)
acc_rf <- sum(y_pred_rf == y_test) / length(y_test)
acc_knn <- sum(y_pred_knn == y_test) / length(y_test)
cat(sprintf("Decision Tree Accuracy: %.2f\n", acc_dt))
cat(sprintf("Random Forest Accuracy: %.2f\n", acc_rf))
cat(sprintf("k-NN Accuracy: %.2f\n", acc_knn))

# Feature Importance Plotting
importance_rf <- as.data.frame(importance(rf_model))
importance_rf$Feature <- rownames(importance_rf)
ggplot(importance_rf, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "forestgreen") +
  coord_flip() +
  labs(title = "Feature Importance - Random Forest", x = "Features", y = "Importance")

# ROC Curves
dt_roc <- multiclass.roc(as.numeric(y_test), as.numeric(y_pred_dt))
plot.roc(dt_roc$rocs[[1]], col = "blue", main = "Decision Tree ROC Curve")
rf_roc <- multiclass.roc(as.numeric(y_test), as.numeric(y_pred_rf))
plot.roc(rf_roc$rocs[[1]], col = "blue", main = "Random Forest ROC Curve")
