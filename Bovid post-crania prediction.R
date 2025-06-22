################################################################
# Load necessary libraries
library(tidyverse)
library(caret)        # For classification evaluation
library(randomForest)
library(e1071)        # For SVM
library(MASS)         # For LDA
library(Amelia)       # For multiple imputation
library(VIM)          # For missing data visualization
library(pROC)         # For AUC-ROC calculations
library(ggpubr)       # For arranging plots
library(FactoMineR)   # For PCA
library(factoextra)   # For PCA visualization
library(ggforce)      # For enhanced ggplot features
library(rstatix)      # For statistical tests
library(car)          # For ANOVA
library(FSA)          # For Dunn's test
library(rcompanion)   # For effect size calculations
library(officer)      # For exporting results to Word
# Data Preparation --------------------------------------------------------
# Load the Bovidae astragalar dataset
bov_ast <- read.csv("Data/bovidae_training_Missing_data.csv", header = TRUE) 
# Data Preparation --------------------------------------------------------
# Visualizing missing data
bov_ast1 <- bov_ast[, 1:3]  # Separate character columns
bov_ast <- bov_ast[, 4:11]   # Select numeric columns
aggr(bov_ast) # Plot missing data pattern
summary(bov_ast) 

# Multiple Imputation -----------------------------------------------------
set.seed(42)
imputed_data1 <- amelia(bov_ast)
completed_data <- imputed_data1$imputations
completed_data_ast <- completed_data %>% as.data.frame()
completed_data_ast1 <- bind_cols(bov_ast1, completed_data_ast)

# Rename imputed columns -----------------------------------------------------
completed_data_ast1 <- completed_data_ast1 %>% 
  mutate(LL = imp1.LL, LI = imp1.LI, LM = imp1.LM, 
         WI = imp1.WI, WD = imp1.WD, TP = imp1.TP,
         TI = imp1.TI, TD = imp1.TD)

# Keep only relevant columns -----------------------------------------------------
completed_data_ast1 <- completed_data_ast1[, 44:51]
completed_data_ast1 <- bind_cols(bov_ast1, completed_data_ast1)

# Split data into training and unknown datasets -----------------------------------------------------
df_training <- completed_data_ast1 %>% slice(1:192)   # First 192 rows for training
df_unknown <- completed_data_ast1 %>% slice(193:n())  # Remaining rows as unknown

# Classification models of bovids' astragali -----------------------------------------------------
# Prepare training data
df_training <- df_training[,-c(1,3)]
data <- df_training
data$Subfamily <- as.factor(data$Subfamily)
data <- data %>% mutate(LI.WI = LI/WI)

# Visualize distribution of Subfamily
ggplot(data, aes(x = Subfamily, fill = Subfamily)) + geom_bar()

# Model Comparison --------------------------------------------------------
set.seed(42)
trainIndex <- sample(2, nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train <- data[trainIndex == 1, ]
test <- data[trainIndex == 2, ]

### 1. RANDOM FOREST MODEL (RF)###
rf <- randomForest(Subfamily ~ ., data = train, ntree = 500, mtry = 1, 
                   importance = TRUE, proximity = TRUE)
rf_pred <- predict(rf, test)
rf_cm <- confusionMatrix(rf_pred, test$Subfamily)

### 2. LINEAR DISCRIMINANT ANALYSIS (LDA) ###
lda_model <- lda(Subfamily ~ ., data = train)
lda_pred <- predict(lda_model, test)$class
lda_cm <- confusionMatrix(lda_pred, test$Subfamily)

### 3. SUPPORT VECTOR MACHINE (SVM) ###
svm_model <- svm(Subfamily ~ ., data = train, kernel = "linear", 
                 cost = 1, scale = TRUE, probability = TRUE)
svm_pred <- predict(svm_model, test,probability = TRUE)
svm_cm <- confusionMatrix(svm_pred, test$Subfamily)

### Enhanced Performance Metrics Function ###
###############################################################
# Enhanced metrics calculation function
# Simplified metrics calculation function
calc_model_metrics <- function(model, cm, test_data, model_name) {
  # Get predicted probabilities
  if (inherits(model, "randomForest")) {
    pred_prob <- predict(model, test_data, type = "prob")
  } else if (inherits(model, "svm")) {
    pred_prob <- attr(predict(model, test_data, probability = TRUE), "probabilities")
    pred_prob <- pred_prob[, levels(test_data$Subfamily)]
  } else if (inherits(model, "lda")) {
    pred_prob <- predict(model, test_data)$posterior
  }
  
  # Multi-class AUC-ROC (One-vs-All approach)
  auc_scores <- sapply(levels(test_data$Subfamily), function(class) {
    suppressMessages({
      roc_obj <- roc(
        response = as.numeric(test_data$Subfamily == class),
        predictor = pred_prob[, class]
      )
    })
    auc(roc_obj)
  })
  
  # Return performance metrics with AUC
  data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Precision = mean(cm$byClass[, "Precision"], na.rm = TRUE),
    Recall = mean(cm$byClass[, "Recall"], na.rm = TRUE),
    F1_Score = mean(cm$byClass[, "F1"], na.rm = TRUE),
    Mean_AUC_ROC = mean(auc_scores),  # Mean AUC across all classes
    Classwise_AUC = I(list(auc_scores)),  # Optional: Store all class-specific AUCs
    stringsAsFactors = FALSE
  )
}

### Model Comparison ###
model_performance <- bind_rows(
  calc_model_metrics(rf, rf_cm, test, "Random Forest"),
  calc_model_metrics(lda_model, lda_cm, test, "LDA"),
  calc_model_metrics(svm_model, svm_cm, test, "SVM")
)

# Print and save results
print(model_performance)
write.csv(model_performance, "Data/Final/model_comparison_auc.csv", row.names = FALSE)

### Class-Specific Performance Visualization ------------------------------
# Function to extract class-wise metrics (corrected)
get_class_metrics <- function(cm, model, test_data) {
  # Calculate AUC scores
  auc_scores <- sapply(levels(test_data$Subfamily), function(class) {
    # Get predicted probabilities for each model type
    if (inherits(model, "randomForest")) {
      pred_prob <- predict(model, test_data, type = "prob")[, class]
    } else if (inherits(model, "svm")) {
      pred_prob <- attr(predict(model, test_data, probability = TRUE), "probabilities")[, class]
    } else if (inherits(model, "lda")) {
      pred_prob <- predict(model, test_data)$posterior[, class]
    }
    
    # Calculate AUC with suppressed messages
    suppressMessages({
      roc_obj <- roc(
        response = as.numeric(test_data$Subfamily == class),
        predictor = pred_prob
      )
      auc(roc_obj)
    })
  })
  
  # Create results data frame with AUC only
  result <- data.frame(
    Subfamily = levels(test_data$Subfamily),
    Precision = cm$byClass[, "Precision"],
    Recall = cm$byClass[, "Recall"],
    F1 = cm$byClass[, "F1"],
    AUC = auc_scores,
    stringsAsFactors = FALSE
  )
  
  return(result)
}
# Generate performance data for Random Forest
rf_perf <- get_class_metrics(rf_cm, rf, test)

# View results
print(rf_perf)
# Visualize
su1<-ggplot(rf_perf, aes(x = reorder(Subfamily, Recall))) +
  geom_col(aes(y = Recall, fill = "Recall"), width = 0.8, alpha = 0.4) +
  geom_col(aes(y = AUC, fill = "AUC"), width = 0.8, alpha = 0.4,
           position = position_nudge(x = 0.2)) +
  geom_text(aes(y = Recall, label = sprintf("%.2f", Recall)),
            position = position_dodge(width = 0.9), hjust = -0.1, size = 3.5) +
  geom_text(aes(y = AUC, label = sprintf("%.2f", AUC)),
            position = position_nudge(x = 0.2), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("Recall" = "#E69F00", "AUC" = "#56B4E9")) +
  coord_flip() +
  labs(title = "Astragali",
       subtitle = "Random Forest Model Test Set Performance",
       x = "Subfamily",
       y = "Score",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1)))
su1
### FOSSIL DATA #########
### Predict Unknown Data --------------------------------------------------
df_unknown <- df_unknown[,-c(2,3)] %>% 
  mutate(LI.WI = LI/WI)

# Predict using Random Forest
predictions <- predict(rf, df_unknown)
df_unknown$Subfamily <- predictions

# Save predictions
write.csv(df_unknown, "Data/Final/unknown_data_predictions_rf_AUC.csv", row.names = FALSE)
#######################################################################
#Random forest to predict habitat preference
# Load dataset with Subfamily to predict the Habitat
data_WSF <- read.csv("Data/bovidae_training_data_habitat.csv",header = TRUE)   #HABITAT (O=1,L=2,H=3,F=4)
set.seed(42)
# Load dataset
data <- data_WSF
data <- data %>%
  mutate(Habitat = case_when(
    Habitat == 1 ~ "O",
    Habitat == 2 ~ "L",
    Habitat == 3 ~ "H",
    Habitat == 4 ~ "F",
    TRUE ~ as.character(Habitat)  # Keep other values unchanged
  ))
data$Habitat <- as.factor(data$Habitat)
data$Subfamily <- as.factor(data$Subfamily)

#data<-data %>% #this is used to asses the accuracy when excluding subfamily as a variable
#select(-Subfamily)
# Data partition
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.8, 0.2))
train <- data[ind == 1, ]
test <- data[ind == 2, ]
### 1. RANDOM FOREST MODEL ###
set.seed(42)
rf <- randomForest(Habitat ~ ., data = train, ntree = 500, mtry = 3, importance = TRUE, proximity = TRUE)

# Predict on test set
rf_pred <- predict(rf, test)
rf_cm <- confusionMatrix(rf_pred, test$Habitat)

### 2. LINEAR DISCRIMINANT ANALYSIS (LDA) ###
lda_model <- lda(Habitat ~ ., data = train)
lda_pred <- predict(lda_model, test)$class
lda_cm <- confusionMatrix(lda_pred, test$Habitat)

### 3. SUPPORT VECTOR MACHINE (SVM) ###
svm_model <- svm(Habitat ~ ., data = train, kernel = "linear", cost = 1, scale = TRUE,
                 probability = TRUE)
svm_pred <- predict(svm_model, test,probability = TRUE)
svm_cm <- confusionMatrix(svm_pred, test$Habitat)
####
# Simplified metrics calculation function 
calc_model_metrics2 <- function(model, cm, test_data, model_name) {
  # Get predicted probabilities 
  pred_prob <- tryCatch({
    if (inherits(model, "randomForest")) {
      predict(model, test_data, type = "prob")
    } else if (inherits(model, "svm")) {
      # Force probability prediction even if model wasn't trained with probability=TRUE
      pred <- predict(model, test_data, probability = TRUE)
      probs <- attr(pred, "probabilities")
      if (is.null(probs)) {
        # Create dummy probabilities based on predicted classes
        probs <- model.matrix(~ . - 1, 
                              data.frame(factor(pred, levels = levels(test_data$Habitat))))
        colnames(probs) <- levels(test_data$Habitat)
      }
      probs[, levels(test_data$Habitat)]
    } else if (inherits(model, "lda")) {
      predict(model, test_data)$posterior
    }
  }, error = function(e) {
    warning("Probability prediction failed, using fallback method: ", e$message)
    # Create simple probability matrix from predicted classes
    preds <- predict(model, test_data)
    model.matrix(~ . - 1, 
                 data.frame(factor(preds, levels = levels(test_data$Habitat))))
  })
  
  # Calculate AUC scores
  auc_scores <- sapply(levels(test_data$Habitat), function(class) {
    suppressMessages({
      roc_obj <- tryCatch({
        roc(
          response = as.numeric(test_data$Habitat == class),
          predictor = pred_prob[, class]
        )
      }, error = function(e) {
        warning(paste("ROC calculation failed for class", class, ":", e$message))
        return(NULL)
      })
    })
    if (!is.null(roc_obj)) auc(roc_obj) else NA
  })
  
  # Return performance metrics
  data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Precision = mean(cm$byClass[, "Precision"], na.rm = TRUE),
    Recall = mean(cm$byClass[, "Recall"], na.rm = TRUE),
    F1_Score = mean(cm$byClass[, "F1"], na.rm = TRUE),
    Mean_AUC_ROC = mean(auc_scores, na.rm = TRUE),
    Classwise_AUC = I(list(auc_scores)),
    stringsAsFactors = FALSE
  )
}
### Model Comparison ###
model_performance2 <- bind_rows(
  calc_model_metrics2(rf, rf_cm, test, "Random Forest"),
  calc_model_metrics2(lda_model, lda_cm, test, "LDA"),
  calc_model_metrics2(svm_model, svm_cm, test, "SVM")
)

# Print and save results
print(model_performance2)
write.csv(model_performance2, "Data/Final/model_comparison_auc_ast_habitat.csv", row.names = FALSE)

#########################
get_class_metrics2 <- function(cm, model, test_data) {
  # Calculate AUC scores
  auc_scores <- sapply(levels(test_data$Habitat), function(class) {
    # Get predicted probabilities for each model type
    if (inherits(model, "randomForest")) {
      pred_prob <- predict(model, test_data, type = "prob")[, class]
    } else if (inherits(model, "svm")) {
      pred_prob <- attr(predict(model, test_data, probability = TRUE), "probabilities")[, class]
    } else if (inherits(model, "lda")) {
      pred_prob <- predict(model, test_data)$posterior[, class]
    }
    
    # Calculate AUC with suppressed messages
    suppressMessages({
      roc_obj <- roc(
        response = as.numeric(test_data$Habitat == class),
        predictor = pred_prob
      )
      auc(roc_obj)
    })
  })
  
  # Create results data frame with AUC only
  result <- data.frame(
    Habitat = levels(test_data$Habitat),
    Precision = cm$byClass[, "Precision"],
    Recall = cm$byClass[, "Recall"],
    F1 = cm$byClass[, "F1"],
    AUC = auc_scores,
    stringsAsFactors = FALSE
  )
  
  return(result)
}


lda_perf <- get_class_metrics2(lda_cm, lda_model, test)

# 3. Visualize Success Rates
su2<-ggplot(lda_perf, aes(x = reorder(Habitat, Recall))) +
  geom_col(aes(y = Recall, fill = "Recall"), width = 0.8, alpha = 0.4) +
  geom_col(aes(y = AUC, fill = "AUC"), width = 0.8, alpha = 0.4,
           position = position_nudge(x = 0.2)) +
  geom_text(aes(y = Recall, label = sprintf("%.2f", Recall)),
            position = position_dodge(width = 0.9), hjust = -0.1, size = 3.5) +
  geom_text(aes(y = AUC, label = sprintf("%.2f", AUC)),
            position = position_nudge(x = 0.2), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("Recall" = "#E69F00", "AUC" = "#56B4E9")) +
  coord_flip() +
  labs(title = "Astragali",
       subtitle = "LDA Model Test Set Performance",
       x = "Habitat",
       y = "Score",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1)))
## FOOSIL DATA #####
# Load unknown data for prediction --------------------------------------------------
unknown_data1 <- df_unknown
unknown_data1$Subfamily<-as.character(unknown_data1$Subfamily)

# Make predictions
unknown_predictions <- predict(lda_model, unknown_data1)$class

# Add predictions to the dataset
unknown_data1$Habitat <- unknown_predictions

# Print the first few predictions
head(unknown_data1)

wieght_prediction<-unknown_data1

# Regression equation that have been used here is adopted from DeGusta and Varba 2003
wieght_prediction$Weight_Kg <- exp(1.343 + 0.060 * wieght_prediction$LL - 
                                     0.068 * wieght_prediction$LI - 
                                     0.007 * wieght_prediction$LM + 
                                     0.047 * wieght_prediction$WI + 
                                     0.025 * wieght_prediction$WD + 
                                     0.015 * wieght_prediction$TP + 
                                     0.044 * wieght_prediction$TI - 
                                     0.001 * wieght_prediction$TD)
wieght_prediction <- wieght_prediction %>%
  mutate(Habitat = case_when(
    Habitat == 1 ~ "O",
    Habitat == 2 ~ "L",
    Habitat == 3 ~ "H",
    Habitat == 4 ~ "F",
    TRUE ~ as.character(Habitat)  # Keep other values unchanged
  ))

write.csv(wieght_prediction, "wg.csv")
# Adding Localities, Units and Sectors
ast_sub<-read.csv("Data/Ast_sub_data.csv",header = T)

final_ast_data<-bind_cols(wieght_prediction,ast_sub)
table(final_ast_data$Habitat,final_ast_data$Subfamily)
unit_u1<-final_ast_data %>% 
  filter(Unit %in% "U1",Sector%in% c("1","2B", "3B","3D", "4B", "4C","4D","5D"))
write.csv(unit_u1,"Data/PCA_ast.csv")
# Save the final report which contain all predictions (Subfamilies and Habitats) based on bovid astragali
write.csv(final_ast_data, "Data/Final/Ast_Sahabi_bovidae_Final.csv", row.names = FALSE)

########################################################
################################################################
#For Proximal phalanges of bovidae #### Same procedure we used with Astragali we use it here
# Load dataset 
bov_ppx <- read.csv("Data/bovidae_training_data_PX.csv",header = TRUE) 
#we will use the function aggr from the package VIM
#to visualize our data
bov_ppx1 <- bov_ppx[, 1:2] # I splited the data to characters
bov_ppx <- bov_ppx[, 3:9] # and for num and I used this one for the analysis
#Visualize first proximal phalanges Data
aggr(bov_ppx)
str(bov_ppx)
summary(bov_ppx) 

set.seed(42) # This for our results so it will not be change if rerun
# Multiple imputation technique using Amelia function
imputed_data1 <- amelia(bov_ppx)
summary(imputed_data1)
plot(imputed_data1)
completed_data<-imputed_data1$imputations
class(completed_data)
completed_data_ast<-completed_data %>% 
  as.data.frame()
completed_data_ast1<-bind_cols(bov_ppx1,completed_data_ast)


completed_data_ast1<-completed_data_ast1 %>% 
  mutate(LM=imp1.LM)  %>% 
  mutate(WP=imp1.WP) %>% 
  mutate(WI=imp1.WI) %>% 
  mutate(WD=imp1.WD) %>% 
  mutate(HP=imp1.HP) %>%
  mutate(HI=imp1.HI) %>%
  mutate(HD=imp1.HD)

str(completed_data_ast1)

completed_data_ast1<-completed_data_ast1%>% 
  mutate(LM.WP=LM/WP)

completed_data_ast1 <- completed_data_ast1[, 38:45]

completed_data_ast1<-bind_cols(bov_ppx1,completed_data_ast1)

#SPLIT THE DATA INTO TRAINING AND UNKOWN
df_training <- completed_data_ast1 %>% slice(1:165)   # First 165 rows for training data
df_unkown <- completed_data_ast1 %>% slice(166:n()) # Remaining rows

df_training<-df_training[,-1]

#Random forest FOR SUBFAMILIES
set.seed(42)

data<-df_training
str(data)
data$Subfamily<-as.factor(data$Subfamily)
table(data$Subfamily)

#data partition
ind<-sample(2,nrow(data),replace = TRUE, prob = c(0.8,0.2))
train<-data[ind==1,]
test<-data[ind==2,]
### 1. RANDOM FOREST MODEL ###
rf <- randomForest(Subfamily ~ ., data = train, ntree = 500, mtry = 1, importance = TRUE, proximity = TRUE)
# Predict on test set
rf_pred <- predict(rf, test)
rf_cm <- confusionMatrix(rf_pred, test$Subfamily)
### 2. LINEAR DISCRIMINANT ANALYSIS (LDA) ###
lda_model <- lda(Subfamily ~ ., data = train)
lda_pred <- predict(lda_model, test)$class
lda_cm <- confusionMatrix(lda_pred, test$Subfamily)
### 3. SUPPORT VECTOR MACHINE (SVM) ###
svm_model <- svm(Subfamily ~ ., data = train, kernel = "linear", cost = 1, scale = TRUE,
                 probability = TRUE)
svm_pred <- predict(svm_model, test,probability = TRUE)
svm_cm <- confusionMatrix(svm_pred, test$Subfamily)
### 4. PERFORMANCE COMPARISON ###
### Model Comparison ###
model_performance3 <- bind_rows(
  calc_model_metrics(rf, rf_cm, test, "Random Forest"),
  calc_model_metrics(lda_model, lda_cm, test, "LDA"),
  calc_model_metrics(svm_model, svm_cm, test, "SVM")
)

# Print and save results
print(model_performance3)
# Save results to CSV
write.csv(model_performance3, "Data/Final/model_comparison_results_PX.csv", row.names = FALSE)
# Generate performance data for Random Forest
LDA_perf2 <- get_class_metrics(lda_cm, lda_model, test)

# View results
print(LDA_perf2)
# Visualize
su3<-ggplot(LDA_perf2, aes(x = reorder(Subfamily, Recall))) +
  geom_col(aes(y = Recall, fill = "Recall"), width = 0.8, alpha = 0.4) +
  geom_col(aes(y = AUC, fill = "AUC"), width = 0.8, alpha = 0.4,
           position = position_nudge(x = 0.2)) +
  geom_text(aes(y = Recall, label = sprintf("%.2f", Recall)),
            position = position_dodge(width = 0.9), hjust = -0.1, size = 3.5) +
  geom_text(aes(y = AUC, label = sprintf("%.2f", AUC)),
            position = position_nudge(x = 0.2), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("Recall" = "#E69F00", "AUC" = "#56B4E9")) +
  coord_flip() +
  labs(title = "Proximal phalanges",
       subtitle = "LDA Model Test Set Performance",
       x = "Subfamily",
       y = "Score",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1)))
su3

# Load unknown data for prediction
unknown_data <- df_unkown

# Predict unknown data classes
predictions <- predict(lda_model, unknown_data)$class

# Save predictions
unknown_data$Subfamily <- predictions
unknown_data
#################################################
#Random forest to predict habitat preference
# Load dataset (replace 'bovidae_fossil_data.csv' with your actual dataset file)
data_pxpWSF <- read.csv("Data/bovidae_training_data_habitat_PX.csv",header = TRUE) #HABITAT (O=1,L=2,H=3,F=4)
str(data_pxpWSF)
data_pxpWSF$Habitat<-as.factor(data_pxpWSF$Habitat)
table(data_pxpWSF$Habitat)
data_pxpWSF<-data_pxpWSF %>% 
  mutate(LM.WP=data_pxpWSF$LM/data_pxpWSF$WP)
# data_pxpWSF<-data_pxpWSF %>%#this is used to asses the accuracy when excluding subfamily as a variable
#   select(-Subfamily)

#data partition
set.seed(42)
ind<-sample(2,nrow(data_pxpWSF),replace = TRUE, prob = c(0.8,0.2))
train<-data_pxpWSF[ind==1,]
test<-data_pxpWSF[ind==2,]

### 1. RANDOM FOREST MODEL ###
rf<-randomForest(Habitat~.,data = train,
                 ntree=300,
                 mtry=1,
                 importance=T,
                 proximity=T)
# Predict on test set
rf_pred <- predict(rf, test)
rf_cm <- confusionMatrix(rf_pred, test$Habitat)

### 2. LINEAR DISCRIMINANT ANALYSIS (LDA) ###
lda_model <- lda(Habitat ~ ., data = train)
lda_pred <- predict(lda_model, test)$class
lda_cm <- confusionMatrix(lda_pred, test$Habitat)

### 3. SUPPORT VECTOR MACHINE (SVM) ###
svm_model <- svm(Habitat ~ ., data = train, kernel = "linear", cost = 1, scale = TRUE,
                 probability = TRUE)
svm_pred <- predict(svm_model, test,probability = TRUE)
svm_cm <- confusionMatrix(svm_pred, test$Habitat)

### 4. PERFORMANCE COMPARISON ###
### Model Comparison ###
model_performance4 <- bind_rows(
  calc_model_metrics2(rf, rf_cm, test, "Random Forest"),
  calc_model_metrics2(lda_model, lda_cm, test, "LDA"),
  calc_model_metrics2(svm_model, svm_cm, test, "SVM")
)
# Print model comparison
print(model_performance4)
#write.csv(model_performance, "Data/Final/model_comparison_results_PX_withoutknown_habitat.csv", row.names = FALSE)

# Save results to CSV
write.csv(model_performance4, "Data/Final/model_comparison_results_PX_withknown_habitat.csv", row.names = FALSE)

SVM_perf <- get_class_metrics2(svm_cm, svm_model, test)
SVM_perf<-SVM_perf %>% 
  mutate(Habitat = case_when(
    Habitat == "1" ~ "O",
    Habitat == "2" ~ "L",
    Habitat == "3" ~ "H",
    Habitat == "4" ~ "F",
    TRUE ~ as.character(Habitat)  # Keep other values unchanged
  ))
# 3. Visualize Success Rates
su4<-ggplot(SVM_perf, aes(x = reorder(Habitat, Recall))) +
  geom_col(aes(y = Recall, fill = "Recall"), width = 0.8, alpha = 0.4) +
  geom_col(aes(y = AUC, fill = "AUC"), width = 0.8, alpha = 0.4,
           position = position_nudge(x = 0.2)) +
  geom_text(aes(y = Recall, label = sprintf("%.2f", Recall)),
            position = position_dodge(width = 0.9), hjust = -0.1, size = 3.5) +
  geom_text(aes(y = AUC, label = sprintf("%.2f", AUC)),
            position = position_nudge(x = 0.2), hjust = -0.1, size = 3.5) +
  scale_fill_manual(values = c("Recall" = "#E69F00", "AUC" = "#56B4E9")) +
  coord_flip() +
  labs(title = "Proximal phalanges",
       subtitle = "SVM Model Test Set Performance",
       x = "Habitat",
       y = "Score",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1, vjust = 1, size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1)))

##########
### FOSSIL DATA ############
# Load unknown data for prediction
unknown_data1 <- unknown_data
unknown_data1$Subfamily<-as.character(unknown_data1$Subfamily)

# Predict unknown data classes
predictions1 <- predict(rf, unknown_data1)

# Save predictions
unknown_data1$Habitat <- predictions1

wieght_prediction<-unknown_data1

# The equation that have been used here is adopted from DeGusta and Varba 2005
wieght_prediction$Weight_Kg <- exp(1.450+0.012*wieght_prediction$LM
                                   +0.046*wieght_prediction$WP-0.004*wieght_prediction$WI+
                                     0.025*wieght_prediction$WD
                                   +0.067*wieght_prediction$HP+
                                     0.022*wieght_prediction$HI-0.032*wieght_prediction$HD)
wieght_prediction <- wieght_prediction %>%
  mutate(Habitat = case_when(
    Habitat == 1 ~ "O",
    Habitat == 2 ~ "L",
    Habitat == 3 ~ "H",
    Habitat == 4 ~ "F",
    TRUE ~ as.character(Habitat)  # Keep other values unchanged
  ))
write.csv(wieght_prediction,"WG2.csv")
# Adding Localites, Units and Sectors
pxp_sub<-read.csv("Data/pxp_sub_data.csv",header = T)

final_pxp_data<-bind_cols(wieght_prediction,pxp_sub)
names(final_pxp_data)[1]<-"Collection.No."

table(final_pxp_data$Habitat,final_pxp_data$Subfamily)
unit_u1_px<-final_pxp_data %>% 
  filter(Unit %in% "U1",Sector%in% c("1","2B", "3B","3D", "4B", "4C","4D","5D"))
write.csv(unit_u1_px,"Data/PCA_pxp.csv")

write.csv(final_pxp_data, "Data/Final/PPX_Sahabi_bovidae_Final.csv", row.names = FALSE)
#Fig4
figure_su1_3 <- ggarrange(su1,su3,
                          labels = c("a", "b"),
                          ncol = 2, nrow = 1,common.legend = T, legend = "top",hjust = -2)
figure_su1_3<-figure_su1_3+
  theme(panel.background = element_rect(fill = "white", color = NA),
        panel.border = element_rect(color = "black", fill = NA, size = 0.5),
        plot.background = element_rect(fill = "white", color = "black", size = 2))

ggsave("figs/R_Output/Fig4.tiff", plot = figure_su1_3, width = 10, height = 5, dpi = 600, compression = "lzw")

# Fig. 6
figure_su2_4 <- ggarrange(su2,su4,
                          labels = c("a", "b"),
                          ncol = 2, nrow = 1,common.legend = T, legend = "top",hjust = -2)
figure_su2_4<-figure_su2_4+
  theme(panel.background = element_rect(fill = "white", color = NA),
        panel.border = element_rect(color = "black", fill = NA, size = 0.5),
        plot.background = element_rect(fill = "white", color = "black", size = 2))

ggsave("Submession file/Fig6.tiff", plot = figure_su2_4, width = 10, height = 5, dpi = 600, compression = "lzw")

#########################
############################################################
######################################################################################################
###############
#Add both data of Astragali and Proximal phalanges into one data frame in order todo some analysis
final_ast_data<-final_ast_data %>% 
  mutate(Element="Astra")
final_pxp_data<-final_pxp_data %>% 
  mutate(Element="Pxp")
pca_ast<-final_ast_data
pca_pxp<-final_pxp_data

table(final_ast_data$Habitat,final_ast_data$Subfamily)
table(final_pxp_data$Habitat,final_pxp_data$Subfamily)

final_ast_data<-final_ast_data[,-2:-10]
final_pxp_data<-final_pxp_data[,-3:-10]
#compare the distribution of specimens into subfamilies between two datasets (final_ast_data and final_pxp_data)
# Contingency tables for Subfamily distribution per Habitat
ast_table <- table(final_ast_data$Subfamily, final_ast_data$Habitat)
pxp_table <- table(final_pxp_data$Subfamily, final_pxp_data$Habitat)

# Print tables
print("Astra Data:")
print(ast_table)

print("Pxp Data:")
print(pxp_table)
# Combine data for comparison (assuming same subfamilies in both datasets)
combined_data <- rbind(
  final_ast_data %>% select(Subfamily, Habitat,Element),
  final_pxp_data %>% select(Subfamily, Habitat,Element)
)

contingency_by_subfamily <- combined_data %>%
  count(Subfamily, Element) %>%
  pivot_wider(names_from = Element, values_from = n, values_fill = 0)

print(contingency_by_subfamily)

# Chi-square test
chi_test <- chisq.test(table(combined_data$Subfamily, combined_data$Element))
print(chi_test)

# Check expected counts (if any <5, consider Fisher's test)
print(chi_test$expected)

# Fisher's exact test as primary test
fisher_test <- fisher.test(table(combined_data$Subfamily, combined_data$Element))
print(fisher_test)

# Output typically includes:
# p-value = 0.0004 (example)
# Alternative hypothesis: two.sided

posthoc_results <- combined_data %>%
  group_by(Subfamily) %>%
  summarise(
    # Create proper 2x2 table: Element vs. a binary outcome
    # Here we compare Element counts against total sample size
    contingency_table = list(
      matrix(c(
        sum(Element == "Astra"),
        sum(Element == "Pxp"),
        nrow(combined_data) - sum(Element == "Astra"),
        nrow(combined_data) - sum(Element == "Pxp")
      ), nrow = 2)
    ),
    # Run Fisher's test with simulation for small samples
    test = list(fisher.test(contingency_table[[1]], simulate.p.value = TRUE, B = 10000)),
    # Extract p-value
    p.value = test[[1]]$p.value,
    .groups = "drop"
  ) %>%
  # Adjust p-values using Holm method
  mutate(p.adj = p.adjust(p.value, method = "holm")) %>%
  # Select final columns
  select(Subfamily, p.value, p.adj)

# Print results
print(posthoc_results)

####################################################################################
####################################################################################
# The distribution of fossil and extant bovid taxa in multidimentional space ########
####################################################################################
######### 1. Astragali ###################################
########################################################
# Read and prepare data
data <- read.csv("Data/PCA_ast2.csv") %>%
  select(LL, LI, LM, WI, WD, TP, TI, TD, LI.WI) %>%
  na.omit()

metadata <- read.csv("Data/PCA_ast2.csv") %>%
  select(Collection.No., Tribe, Habitat, Data) %>%
  filter(complete.cases(read.csv("Data/PCA_ast2.csv") %>% 
                          select(LL, LI, LM, WI, WD, TP, TI, TD, LI.WI)))


# Perform PCA
pca_result <- PCA(data, graph = FALSE)

#  Extract Key Results
# ========================================
# (A) Eigenvectors (variable loadings)
eigenvectors <- pca_result$var$coord
eigenvectors<-eigenvectors %>% 
  as.data.frame()%>%
  rownames_to_column(var = "Variable")  # Convert row names to a column
# (B) Eigenvalues (variance explained)
eigenvalues <- pca_result$eig
eigenvalues<-eigenvalues%>% 
  as.data.frame()%>%
  rownames_to_column(var = "Componenet")  # Convert row names to a column
# (C) Variable contributions (% per PC)
var_contrib <- pca_result$var$contrib
var_contrib<-var_contrib%>% 
  as.data.frame()%>%
  rownames_to_column(var = "Variable")  # Convert row names to a column


# Create base plot without variable vectors
pca_plot <- fviz_pca_ind(pca_result,
                         geom = "point",
                         col.ind = metadata$Habitat,
                         palette = c("#1b9e77", "#d95f02", "#7570b3", "#e7298a","blue", "red", "purple", "yellow"),
                         addEllipses = TRUE,
                         ellipse.type = "confidence",
                         ellipse.level = 0.95,
                         legend.title = "Habitat",
                         title = "PCA of Astragalus Morphology by Habitat",
                         repel = TRUE) +
  theme_minimal() +
  theme(legend.position = "right")

# Remove variable vectors and their labels
pca_plot$layers[[1]] <- NULL  # Removes variable arrows
pca_plot$labels$colour <- "Habitat"  # Fix legend title

# Add convex hulls by habitat (alternative to ellipses)
pca_plot <- pca_plot +
  geom_mark_hull(
    aes(fill = metadata$Habitat,
        color = metadata$Habitat),
    alpha = 0.1,
    expand = unit(2, "mm"),
    show.legend = FALSE  # Avoid duplicate legend entries
  )

# Final adjustments
pca_plot <- pca_plot +
  labs(color = "Habitat") +
  guides(fill = "none")  # Remove fill legend

print(pca_plot)

# Load additional data (replace with your second dataset path)
As_Sahabi <- pca_ast %>% 
  select(LL, LI, LM, WI, WD, TP, TI, TD, LI.WI, Subfamily,Habitat,Collection.No.) %>%
  na.omit()

# Predict PCA coordinates
new_pca <- predict(pca_result, newdata = select(As_Sahabi, -Subfamily))

# (D) Individual specimen scores
ind_scores <- new_pca$coord
ind_scores<-ind_scores %>% 
  as.data.frame() 
# Save all results as an R object for later use
pca_summary <- list(
  Eigenvectors = eigenvectors,
  Eigenvalues = eigenvalues,
  Variable_Contributions = var_contrib,
  Individual_Scores = ind_scores
)
#Export to Word (docx)
# ========================================
# Create a Word document
doc <- officer::read_docx()

# Add title
doc <- doc %>% 
  body_add_par("PCA Results Summary", style = "heading 1")

# Add Eigenvalues table
doc <- doc %>% 
  body_add_par("Eigenvalues (Variance Explained)", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary$Eigenvalues), style = "table_template")

# Add Eigenvectors table
doc <- doc %>% 
  body_add_par("Eigenvectors (Variable Loadings)", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary$Eigenvectors), style = "table_template")
# Add Variable_Contributions table
doc <- doc %>% 
  body_add_par("Variable_Contributions", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary$Variable_Contributions), style = "table_template")
# # Add Individual_Scores table
doc <- doc %>% 
  body_add_par("As-Sahabi Individual_Scores", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary$Individual_Scores), style = "table_template")



# Save Word file
print(doc, target = "Data/Final/Ast_pca_summary.docx")


#Convert to plottable format
new_points <- data.frame(
  PC1 = new_pca$coord[,1],
  PC2 = new_pca$coord[,2],
  Source = "New Data"  # Identifier for legend
)

# Get original PCA coordinates
original_points <- data.frame(
  PC1 = pca_result$ind$coord[,1],
  PC2 = pca_result$ind$coord[,2],
  Habitat = metadata$Habitat,
  Source = "Original Data"
)
# Create plotting data with subfamily info AND specimen numbers
new_points <- data.frame(
  PC1 = new_pca$coord[,1],
  PC2 = new_pca$coord[,2],
  As_Sahabi = As_Sahabi$Subfamily,
  Collection_No = As_Sahabi$Habitat,  # Add specimen numbers
  loc_num = As_Sahabi$Collection.No.,
  Source = "As_Sahabi"
)
write.csv(new_points,"pc1_ast_results.csv")
habitat_colors<- c("darkgreen", "lightgreen", "blue", "orange")
habitat_colors2<- c("green", "orange", "blue", "red", "purple", "yellow")
# Then modify your plot code to add text labels:
pca_plot <- ggplot() +
  # Original data (habitat clusters)
  geom_point(data = original_points,
             aes(PC1, PC2, color = Habitat),
             size = 2, alpha = 0.4) +
  
  # New data points (colored by subfamily)
  geom_point(data = new_points,
             aes(PC1, PC2, fill = As_Sahabi, shape = As_Sahabi),
             size = 3, stroke = 0.8) +
  
  # Add text labels for specimen numbers
  geom_text(data = new_points,
            aes(PC1, PC2, label = Collection_No),
            hjust = -0.5, vjust = -0.8, size = 2.5) +
  
  # Habitat ellipses
  stat_ellipse(data = original_points,
               aes(PC1, PC2, color = Habitat),
               type = "t", level = 0.95, linewidth = 0.8) +
  
  # Cosmetic settings
  scale_fill_manual(values = habitat_colors2) +
  scale_color_manual(values = habitat_colors) +
  scale_shape_manual(values = c(21,22,23,24,25,20,19)) +
  labs(title = "Astragali",
       x = paste0("PC1 (", round(pca_result$eig[1,2],1), "%)"),
       y = paste0("PC2 (", round(pca_result$eig[2,2],1), "%)"),
       shape="As-Sahabi subfamilies", color="Extant African Habitat",fill="As-Sahabi subfamilies") +
  theme_minimal() +
  theme(axis.title = element_text(size = 14,face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1,size = 12),
        axis.text.y =  element_text(angle = 0, hjust = 1, vjust = 1,size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
        legend.text = element_text(size = 12))

# Extract variable coordinates (loadings) from PCA results
var_coords <- as.data.frame(pca_result$var$coord[, 1:2])  # PC1 and PC2
var_coords$Variable <- rownames(var_coords)  # Add variable names

# Scale factor for arrow length (adjust as needed)
scale_factor <- 5  # Increase to make arrows longer

# Add variable arrows to your existing plot
pca_plot_with_vars <- pca_plot +
  # Add variable arrows (scaled for visibility)
  geom_segment(
    data = var_coords,
    aes(x = 0, y = 0, xend = scale_factor * Dim.1, yend = scale_factor * Dim.2),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "darkred",
    linewidth = 0.7
  ) +
  # Add variable labels (repel to avoid overlap)
  ggrepel::geom_text_repel(
    data = var_coords,
    aes(x = scale_factor * Dim.1, y = scale_factor * Dim.2, label = Variable),
    color = "darkred",
    size = 4,
    fontface = "bold"
  )

# Print the enhanced plot
print(pca_plot_with_vars)
#######################################
######### 2. Proximal phalenges #######
#The distribution of fossil and extant bovid taxa in multidimentional space
# Read and prepare data
data1 <- read.csv("Data/PCA_pxp2.csv") %>%
  select(LM, WP, WI,WD, HP, HI, HD, LM.WP) %>%
  na.omit()

metadata1 <- read.csv("Data/PCA_pxp2.csv") %>%
  select(Collection.No., Subfamily, Habitat, Data) %>%
  filter(complete.cases(read.csv("Data/PCA_pxp2.csv") %>% 
                          select(LM, WP, WI,WD, HP, HI, HD, LM.WP)))

# Perform PCA
pca_result1 <- PCA(data1, graph = FALSE)
#  Extract Key Results
# ========================================
# (A) Eigenvectors (variable loadings)
eigenvectors1 <- pca_result1$var$coord
eigenvectors1<-eigenvectors1 %>% 
  as.data.frame()%>%
  rownames_to_column(var = "Variable")  # Convert row names to a column
# (B) Eigenvalues (variance explained)
eigenvalues1 <- pca_result1$eig
eigenvalues1<-eigenvalues1%>% 
  as.data.frame()%>%
  rownames_to_column(var = "Componenet")  # Convert row names to a column
# (C) Variable contributions (% per PC)
var_contrib1 <- pca_result1$var$contrib
var_contrib1<-var_contrib1%>% 
  as.data.frame()%>%
  rownames_to_column(var = "Variable")  # Convert row names to a column

# Create base plot without variable vectors
pca_plot2 <- fviz_pca_ind(pca_result1,
                          geom = "point",
                          col.ind = metadata1$Habitat,
                          palette = c("#1b9e77", "#d95f02", "#7570b3", "#e7298a"),
                          addEllipses = TRUE,
                          ellipse.type = "confidence",
                          ellipse.level = 0.95,
                          legend.title = "Habitat",
                          title = "PCA of Proximal Phalenges Morphology by Habitat",
                          repel = TRUE) +
  theme_minimal() +
  theme(legend.position = "right")

# Remove variable vectors and their labels
pca_plot2$layers[[1]] <- NULL  # Removes variable arrows
pca_plot2$labels$colour <- "Habitat"  # Fix legend title

# Add convex hulls by habitat (alternative to ellipses)
pca_plot2 <- pca_plot2 +
  geom_mark_hull(
    aes(fill = metadata1$Habitat,
        color = metadata1$Habitat),
    alpha = 0.1,
    expand = unit(2, "mm"),
    show.legend = FALSE  # Avoid duplicate legend entries
  )

# Final adjustments
pca_plot2 <- pca_plot2 +
  labs(color = "Habitat") +
  guides(fill = "none")  # Remove fill legend

print(pca_plot2)

# Load additional data (replace with your second dataset path)
As_Sahabi1 <- pca_pxp %>% 
  select( LM, WP, WI,WD, HP, HI, HD, LM.WP, Subfamily,Habitat,Collection.No.) %>%
  na.omit()

# Predict PCA coordinates
new_pca1 <- predict(pca_result1, newdata = select(As_Sahabi1, -Subfamily))
# (D) Individual specimen scores
ind_scores1 <- new_pca1$coord
ind_scores1<-ind_scores1 %>% 
  as.data.frame() 
# Save all results as an R object for later use
pca_summary1 <- list(
  Eigenvectors1 = eigenvectors1,
  Eigenvalues1 = eigenvalues1,
  Variable_Contributions1 = var_contrib1,
  Individual_Scores1 = ind_scores1
)
#Export to Word (docx)
# ========================================
# Create a Word document
doc1 <- officer::read_docx()

# Add title
doc1 <- doc1 %>% 
  body_add_par("PCA Results Summary of As-Sahabi Proximal Phalanges", style = "heading 1")

# Add Eigenvalues table
doc1 <- doc1 %>% 
  body_add_par("Eigenvalues (Variance Explained)", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary1$Eigenvalues1), style = "table_template")

# Add Eigenvectors table
doc1 <- doc1 %>% 
  body_add_par("Eigenvectors (Variable Loadings)", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary1$Eigenvectors1), style = "table_template")
# Add Variable_Contributions table
doc1 <- doc1 %>% 
  body_add_par("Variable_Contributions", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary1$Variable_Contributions1), style = "table_template")
# # Add Individual_Scores table
doc1 <- doc1 %>% 
  body_add_par("As-Sahabi Individual_Scores", style = "heading 2") %>%
  body_add_table(as.data.frame(pca_summary1$Individual_Scores1), style = "table_template")



# Save Word file
print(doc1, target = "Data/Final/Pxp_pca_summary.docx")


# Get original PCA coordinates
original_points1 <- data.frame(
  PC1 = pca_result1$ind$coord[,1],
  PC2 = pca_result1$ind$coord[,2],
  Habitat = metadata1$Habitat,
  Source = "Original Data"
)
# Create plotting data with subfamily info
new_points1 <- data.frame(
  PC1 = new_pca1$coord[,1],
  PC2 = new_pca1$coord[,2],
  As_Sahabi = As_Sahabi1$Subfamily,
  Collection_No = As_Sahabi1$Habitat,  # Add specimen numbers
  loc_num = As_Sahabi1$Collection.No.,
  Source = "As_Sahabi"
)

write.csv(new_points1,"pc1_pxp_results.csv")



habitat_colors<- c("darkgreen", "lightgreen", "blue", "orange")
habitat_colors2<- c("green", "orange", "blue", "red", "purple", "yellow")
# Create combined plot with subfamily differentiation
pca_plot2 <- ggplot() +
  # Original data (habitat clusters)
  geom_point(data = original_points1,
             aes(PC1, PC2, color = Habitat),
             size = 2, alpha = 0.4) +
  
  # New data points (colored by subfamily)
  geom_point(data = new_points1,
             aes(PC1, PC2, fill = As_Sahabi, shape = As_Sahabi),
             size = 3, stroke = 0.8) +
  # Add text labels for specimen numbers
  geom_text(data = new_points1,
            aes(PC1, PC2, label = Collection_No),
            hjust = -0.5, vjust = -0.8, size = 2.5) +
  # Habitat ellipses
  stat_ellipse(data = original_points1,
               aes(PC1, PC2, color = Habitat),
               type = "t", level = 0.95, linewidth = 0.8) +
  
  # Cosmetic settings
  scale_color_manual(values = habitat_colors) +
  scale_fill_manual(values = habitat_colors2) +  # Define your subfamily colors
  scale_shape_manual(values = c(21,22,23,24,25,20,19)) +
  labs(title = "Proximal Phalanges",
       x = paste0("PC1 (", round(pca_result1$eig[1,2],1), "%)"),
       y = paste0("PC2 (", round(pca_result1$eig[2,2],1), "%)"),
       shape="As-Sahabi subfamilies", color="Extant African Habitat",fill="As-Sahabi subfamilies") +
  theme_minimal()+
  theme(axis.title = element_text(size = 14,face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1,size = 12),
        axis.text.y =  element_text(angle = 0, hjust = 1, vjust = 1,size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
        legend.text = element_text(size = 12))

# Extract variable coordinates (loadings) from PCA results
var_coords <- as.data.frame(pca_result1$var$coord[, 1:2])  # PC1 and PC2
var_coords$Variable <- rownames(var_coords)  # Add variable names

# Scale factor for arrow length (adjust as needed)
scale_factor <- 5  # Increase to make arrows longer

# Add variable arrows to your existing plot
pca_plot_with_vars2 <- pca_plot2 +
  # Add variable arrows (scaled for visibility)
  geom_segment(
    data = var_coords,
    aes(x = 0, y = 0, xend = scale_factor * Dim.1, yend = scale_factor * Dim.2),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "darkred",
    linewidth = 0.7
  ) +
  # Add variable labels (repel to avoid overlap)
  ggrepel::geom_text_repel(
    data = var_coords,
    aes(x = scale_factor * Dim.1, y = scale_factor * Dim.2, label = Variable),
    color = "darkred",
    size = 4,
    fontface = "bold"
  )

# Print the enhanced plot
print(pca_plot_with_vars2)


figure_pca <- ggarrange(pca_plot_with_vars,pca_plot_with_vars2,
                        labels = c("a", "b"),
                        ncol = 2, nrow = 1,common.legend = T, legend = "top",hjust = -2)
fig_pca<-figure_pca+
  theme(panel.background = element_rect(fill = "white", color = NA),
        panel.border = element_rect(color = "black", fill = NA, size = 0.5),
        plot.background = element_rect(fill = "white", color = "black", size = 2))

fig_pca
ggsave("Submession file/Fig_7.tiff", plot = fig_pca, width = 12, height = 8, dpi = 600, compression = "lzw")
#####################################################################################
####### nonparametric_analysis FOR COMPARISION BETWEEN SUBFAMILIES AND HABITATS#####
######  ASTRAGALI #############
pca_data1 <- read_csv("pc1_ast_results.csv") %>%
  select(-1) %>% # Remove the first column (row numbers)
  mutate(As_Sahabi = as.factor(As_Sahabi),
         Collection_No = as.factor(Collection_No),
         loc_num = as.factor(loc_num))

####################################################################################
####### nonparametric_analysis FOR COMPARISION BETWEEN SUBFAMILIES AND HABITATS#####
######  PROXIMAL PHALANGES #############
pca_data2 <- read_csv("pc1_pxp_results.csv") %>%
  select(-1) %>% # Remove the first column (row numbers)
  mutate(As_Sahabi = as.factor(As_Sahabi),
         Collection_No = as.factor(Collection_No),
         loc_num = as.factor(loc_num))


# 5. Statistical Analysis 1 SUBFAMILY------------------------------------------------------
########################
#check for normality inorder to chose the test
# Q-Q plot for PC1 by As_Sahabi
ggplot(pca_data1, aes(sample = PC2, color = As_Sahabi)) +
  stat_qq() +
  stat_qq_line() +
  facet_wrap(~As_Sahabi, scales = "free") +
  labs(title = "Q-Q Plot for PC1 by Subfamily")

# Histograms
ggplot(pca_data1, aes(x = PC1, fill = As_Sahabi)) +
  geom_histogram(bins = 15, alpha = 0.7) +
  facet_wrap(~As_Sahabi, scales = "free") +
  labs(title = "Distribution of PC1 by Subfamily")
# Proceed with ANOVA only if normality assumptions are met
# (If normality is violated, consider non-parametric alternatives like Kruskal-Wallis)
# ================================================
# Analysis for Collection_No grouping
# ================================================
# Function to perform and print non-parametric analysis
run_nonparametric_analysis <- function(data, dv, group_var, results_file = NULL) {
  if (!is.null(results_file)) sink(results_file, append = TRUE)
  
  cat("\n=== Kruskal-Wallis Test for", dv, "by", group_var, "===\n")
  kw_test <- kruskal.test(reformulate(group_var, dv), data = data)
  print(kw_test)
  
  cat("\n=== Dunn's Post-Hoc Test for", dv, "===\n")
  dunn_test <- dunnTest(reformulate(group_var, dv), data = data, method = "bh")
  print(dunn_test)
  
  if (!is.null(results_file)) sink()
}

# ================================================
# Analysis for As_Sahabi grouping Astragali
# ================================================
#1 subfamily
# PC1 Analysis
run_nonparametric_analysis(pca_data1, "PC1", "As_Sahabi")
# PC2 Analysis
run_nonparametric_analysis(pca_data1, "PC2", "As_Sahabi")

# Save results for As_Sahabi
sink("nonparametric_results_As_Sahabi_asra_subfamily.txt")
run_nonparametric_analysis(pca_data1, "PC1", "As_Sahabi")
run_nonparametric_analysis(pca_data1, "PC2", "As_Sahabi")
sink()

# 2 habitat
# PC1 Analysis
run_nonparametric_analysis(pca_data1, "PC1", "Collection_No")
# PC2 Analysis
run_nonparametric_analysis(pca_data1, "PC2", "Collection_No")

# Save results for As_Sahabi
sink("nonparametric_results_As_Sahabi_asra_habitat.txt")
run_nonparametric_analysis(pca_data1, "PC1", "Collection_No")
run_nonparametric_analysis(pca_data1, "PC2", "Collection_No")
sink()

# ================================================
# Analysis for As_Sahabi grouping Proximal Phalanges
# ================================================
#1 subfamily
# PC1 Analysis
run_nonparametric_analysis(pca_data2, "PC1", "As_Sahabi")
# PC2 Analysis
run_nonparametric_analysis(pca_data2, "PC2", "As_Sahabi")

# Save results for As_Sahabi
sink("nonparametric_results_As_Sahabi_pxp_subfamily.txt")
run_nonparametric_analysis(pca_data2, "PC1", "As_Sahabi")
run_nonparametric_analysis(pca_data2, "PC2", "As_Sahabi")
sink()


# 2 habitat
# PC1 Analysis
run_nonparametric_analysis(pca_data2, "PC1", "Collection_No")
# PC2 Analysis
run_nonparametric_analysis(pca_data2, "PC2", "Collection_No")

# Save results for As_Sahabi
sink("nonparametric_results_As_Sahabi_pxp_habitat.txt")
run_nonparametric_analysis(pca_data2, "PC1", "Collection_No")
run_nonparametric_analysis(pca_data2, "PC2", "Collection_No")
sink()


##BOXPLOT FOR EXPLOARING THE DATA
your_data<-read.csv("Data/Final/Ast_Sahabi_bovidae_Final.csv")
# Create the boxplot
ggplot(your_data, aes(x = Subfamily, y = LI.WI, fill = Subfamily)) +
  geom_boxplot(alpha = 0.8, outlier.shape = NA) +  # Hide outliers to avoid double plotting
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +  # Show all points
  scale_fill_brewer(palette = "Set1") +
  labs(title = "LI.WI Ratio Distribution by Subfamily",
       x = "Subfamily",
       y = "LI.WI Ratio",
       fill = "Subfamily") +
  theme_minimal() +
  theme(legend.position = "right",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))


ggplot(your_data, aes(x = Habitat, y = LI.WI, fill = Habitat)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.4) +
  stat_compare_means(method = "anova", label.y = max(your_data$LI.WI) * 1.1) +  # Add ANOVA p-value
  stat_compare_means(label = "p.signif", method = "t.test", 
                     ref.group = ".all.", hide.ns = TRUE) +  # Compare each group to mean
  scale_fill_brewer(palette = "Set2") +
  labs(#title = "LI.WI Ratio by Subfamily with Statistical Comparisons",
    x = "Subfamily",
    y = "LI.WI Ratio") +
  theme_classic() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

##################################################
#################################################
final_Full_data<-bind_rows(final_ast_data,final_pxp_data)
table(final_Full_data$Unit,final_Full_data$Element)

final_Full_data_u1<-final_Full_data%>% 
  filter(Unit %in% "U1")
table(final_Full_data_u1$Element,final_Full_data_u1$Habitat,final_Full_data_u1$Subfamily)

habitat_colors<- c("darkgreen", "lightgreen", "blue", "orange")
data_summary <- final_Full_data_u1 %>%
  group_by(Element, Subfamily, Habitat) %>%
  summarise(Count = n(), .groups = "drop")  # Count occurrences

fig_3<-ggplot(final_Full_data_u1, aes(x = Subfamily, fill = Habitat)) +
  geom_bar(position = "fill")+
  theme_classic() +
  labs(#title = "Bovidae Elements Distribution Across Habitats",
    x = "Subfamily",
    y = "Relative abundance",
    fill = "Habitat") +
  facet_wrap(~Element)+
  scale_fill_manual(values = habitat_colors) +  # Define your subfamily colors
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5,size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1,size = 12),
        axis.title.x = element_text(vjust = 1,size = 14,face = "bold"),
        axis.title.y = element_text(vjust = 1,size = 14,face = "bold"),
        legend.text = element_text(angle = 0, hjust = 1,size = 12),
        legend.title =  element_text(angle = 0, hjust = 1,size = 14),
        strip.text = element_text(size = 14,face = "bold"),
        strip.background = element_blank(),
        legend.position = "top")
fig_3
ggsave("Submession file/Fig_2.tiff", plot = fig_3, width = 12, height = 8, dpi = 600, compression = "lzw")
######
# ============================================
# Paired Sample Analysis COMAPRING THIS STUDY WITH PREVIOS STUDIES OF BOVIDAE
# ============================================
df <- tribble(
  ~subfamily,       ~This_study, ~Previous_studies,
  "Alcelaphinae",     5.652173913,     1.612903226,
  "Antilopinae",     59.13043478,    59.67741935,
  "Bovinae",          20.43478261,     22.58064516,
  "Hippotraginae",    8.260869565,     3.225806452,
  "Reduncinae", 8.260869565,12.90322581
) %>%
  mutate(diff = This_study - Previous_studies)

# --------------------------------------------
# 2. Reshape data for ggplot barplot
# --------------------------------------------
df_long <- df %>%
  pivot_longer(cols = c(This_study, Previous_studies),
               names_to = "source",
               values_to = "value")

# --------------------------------------------
# 3. Barplot comparison
# --------------------------------------------
# Professional Barplot
fig_5<-ggplot(df_long, aes(x = subfamily, y = value, fill = source)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = c("This_study" = "#1f77b4",  # blue
                               "Previous_studies" = "#ff7f0e"),
                    labels = c(  "Previous studies","This study")) +  
  labs(
    #title = "Specimen Counts by Subfamily",
    #subtitle = "Comparison Between This Study and Previous Studies",
    x = "Subfamily",
    y = "Specimen Percentage %",
    fill = "Data"
  ) +
  theme_classic(base_size = 14) +
  theme(axis.text.x = element_text(angle = 0, hjust = 1,size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1,size = 12),
        axis.title.x = element_text(vjust = 1,size = 14,face = "bold"),
        axis.title.y = element_text(vjust = 1,size = 14,face = "bold"),
        legend.text = element_text(angle = 0, hjust = 1,size = 12),
        legend.title =  element_text(angle = 0, hjust = 1,size = 14),
        legend.position = "top")

fig_5
ggsave("figs/R_Output/Fig5.tiff", plot = fig_5, width = 10, height = 8, dpi = 600, compression = "lzw")
# --------------------------------------------
#  QQ Plot of Differences
# --------------------------------------------
ggqqplot(df$diff, title = "QQ Plot of Paired Differences")
# --------------------------------------------
#  Shapiro-Wilk Normality Test
# --------------------------------------------
cat("Shapiro-Wilk Test for Normality:\n")
print(shapiro.test(df$diff))
# --------------------------------------------
#  Paired t-test
# --------------------------------------------
cat("\nPaired t-test:\n")
print(t.test(df$This_study, df$Previous_studies, paired = TRUE))
#########################################################
data_summary1 <- final_Full_data_u1 %>%
  group_by(Unit, Subfamily, Habitat) %>%
  summarise(Count = n(), .groups = "drop")  # Count occurrences

fig7<-ggplot(data_summary1, aes(x = Subfamily, y = Count, fill = Habitat)) +
  geom_bar(stat = "identity", position = "dodge") +  
  theme_classic() +
  labs(#title = "Bovidae Elements Distribution Across Habitats",
    x = "Subfamily",
    y = "Specimen Count (N)",
    fill = "Habitat") +
  scale_fill_manual(values = habitat_colors) +  # Define your subfamily colors
  theme(axis.text.x = element_text(angle = 30, hjust = 1,size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1,size = 12),
        axis.title.x = element_text(vjust = 0.5,size = 14,face = "bold"),
        axis.title.y = element_text(vjust = 1,size = 14,face = "bold"),
        legend.text = element_text(angle = 0, hjust = 1,size = 12),
        legend.title =  element_text(angle = 0, hjust = 1,size = 14),
        legend.position = "top")
fig7
fig7_2<-ggplot(final_Full_data_u1, aes(x = Subfamily, fill = Habitat)) +
  geom_bar(position = "fill") +
  labs(#title = "Proportion of Habitat Types by Subfamily",
    x = "Subfamily",
    y = "Relative abundance") +
  theme_minimal() +
  scale_fill_manual(values = habitat_colors) +  # Define your subfamily colors
  theme(axis.text.x = element_text(angle = 30, hjust = 1,size = 12),
        axis.text.y = element_text(angle = 0, hjust = 1,size = 12),
        axis.title.x = element_text(vjust = 0.5,size = 14,face = "bold"),
        axis.title.y = element_text(vjust = 1,size = 14,face = "bold"),
        legend.text = element_text(angle = 0, hjust = 1,size = 12),
        legend.title =  element_text(angle = 0, hjust = 1,size = 14),
        legend.position = "top")

figure_7 <- ggarrange(fig7,fig7_2,
                      labels = c("a", "b"),
                      ncol = 2, nrow = 1,common.legend = T, legend = "top",hjust = -0.5)
figure_7<-figure_7+
  theme(panel.background = element_rect(fill = "white", color = NA),
        panel.border = element_rect(color = "black", fill = NA, size = 0.5),
        plot.background = element_rect(fill = "white", color = "black", size = 2))
ggsave("Submession file/Fig_6.tiff", plot = figure_7, width = 12, height = 6, dpi = 600, compression = "lzw")

########################################
final_Full_data2<-final_Full_data_u1 %>% 
  filter(Locality %in% c("P5","P11", "P14","P16","P17","P24", "P25", "P28",
                         "P30","P32","P33","P34","P36","P47","P49", "P60",
                         "P61","P62", "P63","P65","P99","P103","P106"))

##################################
# 1. Calculate total Open (O) and Forest (F) counts per locality
locality_order <- final_Full_data2 %>%
  group_by(Locality, Habitat) %>%
  summarise(Count = n(), .groups = "drop") %>%
  pivot_wider(names_from = Habitat, values_from = Count, values_fill = 0) %>%
  mutate(O_ratio = O / (O + F)) %>%  # Calculate proportion of Open habitats
  arrange(desc(O_ratio)) %>%          # Sort from most O to most F
  pull(Locality)                      # Extract ordered locality names

# 2. Create the summary data (now with ordered localities)
data_summary2 <- final_Full_data2 %>%
  mutate(Sector = factor(Locality, levels = locality_order)) %>%
  group_by(Locality, Subfamily, Habitat) %>%
  summarise(Count = n(), .groups = "drop")

# 3. Enhanced visualization
# Prepare contingency table for chi-square test
contingency_table <- final_Full_data2 %>%
  count(Locality, Habitat) %>%
  pivot_wider(names_from = Habitat, values_from = n, values_fill = 0) %>%
  column_to_rownames("Locality") %>%
  as.matrix()

# Check sample size
if(sum(contingency_table) < 20) {
  warning("Total sample size <20 - chi-square may not be appropriate")
}

# Perform chi-square test with simulation if needed
chi_test <- if(sum(chi_test$expected < 5) > 0.2 * length(chi_test$expected)) {
  chisq.test(contingency_table, simulate.p.value=TRUE, B=2000)
} else {
  chisq.test(contingency_table)
}

# Perform chi-square test
chi_test <- chisq.test(contingency_table)

# Print results
cat("\nChi-Square Test of Habitat Distribution Across Localities:\n")
print(chi_test)

# Check expected frequencies (for assumption verification)
cat("\nExpected Frequencies:\n")
print(round(chi_test$expected, 1))

# Check if >20% of cells have expected counts <5
problematic_cells <- sum(chi_test$expected < 5) / length(chi_test$expected)
if(problematic_cells > 0.2) {
  warning("Chi-square assumptions violated - consider Fisher's exact test")
}
# Fisher's Exact Test (recommended for sparse data)
fisher_test <- fisher.test(contingency_table, simulate.p.value = TRUE, B = 10000)
cat("\nFisher's Exact Test (with simulation):\n")
print(fisher_test)

# Calculate effect size (Cramer's V)
cramers_v <- assocstats(contingency_table)$cramer
cat(sprintf("\nEffect Size (Cramer's V): %.2f\n", cramers_v))

# Pairwise comparisons with Bonferroni correction
pairwise_chisq <- pairwise_chisq_gof_test(as.data.frame(contingency_table))
print(pairwise_chisq, n = Inf)  # Show all comparisons

fig5<-ggplot(data_summary2, aes(x = Locality, y = Count, fill = Habitat)) +
  geom_bar(stat = "identity", position = position_dodge(preserve = "single"), width = 0.7) +
  scale_fill_manual(
    values = c("O" = "orange",  # Orange for Open
               "L" = "lightgreen",  # Light blue for Light woodland
               "H" = "blue",  # Green for Heavy woodland
               "F" = "darkgreen"), # Dark blue for Forest
    labels = c( "F", "H", "L","O")
  ) +
  labs(
    #title = "Bovidae Habitat Distribution Across Localities",
    subtitle = "Localities ordered from most Open (O) to most Forest (F) dominated",
    x = "Locality",
    y = "Specimen Count (N)",
    #caption = "Habitat classification: O=Open, L=Light woodland, H=Heavy woodland, F=Forest"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40")
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) + # Pad top only # Your existing plot code
  labs(subtitle = sprintf("Fisher-test: p-value=%.3f, Cramer's V=%.2f",
                          fisher_test$p.value,
                          cramers_v)) 
fig5
#ggsave("figs/R_Output/Appindix 5.tiff", plot = fig5, width = 9, height = 6, dpi = 600, compression = "lzw")
ggsave("figs/R_Output/Fig9.tiff", plot = fig5, width = 9, height = 6, dpi = 600, compression = "lzw")
##############################################################################
###########################################################
########### Weight distibutions############################
final_Full_data3<-final_Full_data %>% 
  filter(Unit %in% "U1")

grouped_summary <- final_Full_data3 %>%
  group_by(Habitat) %>%
  summarise(
    Mean = mean(Weight_Kg, na.rm = TRUE),
    Median = median(Weight_Kg, na.rm = TRUE),
    SD = sd(Weight_Kg, na.rm = TRUE),
    Min = min(Weight_Kg, na.rm = TRUE),
    Max = max(Weight_Kg, na.rm = TRUE),
    IQR = IQR(Weight_Kg, na.rm = TRUE),
    N = sum(!is.na(Weight_Kg)),
    .groups = 'drop'
  )
grouped_summary
write.csv(grouped_summary,"grouped_summary.csv")
# 1. Enhanced Visualization ----------------------------------------------------
# Calculate the mean weight across all localities
final_Full_data2<-final_Full_data_u1 %>% 
  filter(Locality %in% c("P5","P11", "P14","P16","P17","P24", "P25", "P28",
                         "P30","P32","P33","P34","P36","P47","P49", "P60",
                         "P61","P62", "P63","P65","P99","P103","P106"))

final_Full_data2<-final_Full_data2 %>% 
  filter(Weight_Kg < 300)
grouped_summary <- final_Full_data2 %>%
  group_by(Habitat) %>%
  summarise(
    Mean = mean(Weight_Kg, na.rm = TRUE),
    Median = median(Weight_Kg, na.rm = TRUE),
    SD = sd(Weight_Kg, na.rm = TRUE),
    Min = min(Weight_Kg, na.rm = TRUE),
    Max = max(Weight_Kg, na.rm = TRUE),
    IQR = IQR(Weight_Kg, na.rm = TRUE),
    N = sum(!is.na(Weight_Kg)),
    .groups = 'drop'
  )
print(grouped_summary, n=42)
write.csv(grouped_summary,"Data/Final/weight_distrib_13_loc.csv")
mean_weight <- mean(final_Full_data3$Weight_Kg, na.rm = TRUE)
mean_weight
weight_plot <- ggplot(final_Full_data2, 
                      aes(x = reorder(Locality, Weight_Kg, median), 
                          y = Weight_Kg, 
                          fill = Habitat)) +
  geom_boxplot(width = 0.7, 
               alpha = 0.8,
               outlier.shape = 21,
               outlier.fill = "red",
               outlier.size = 2) +
  geom_jitter(width = 0.15, 
              alpha = 0.3, 
              size = 1.5) +
  geom_hline(yintercept = mean_weight, 
             linetype = "dashed", 
             color = "red", 
             size = 1) +  # Add dashed line at mean
  scale_fill_manual(
    values = c("O" = "orange", 
               "L" = "blue",  
               "H" = "lightgreen", 
               "F" = "darkgreen"))+
  labs(#title = "Body Weight Distribution Across Localities",
    #subtitle = "Grouped by Habitat Type",
    x = "Locality (Ordered by Median Weight)",
    y = "Estimated Weight (kg)",
    #caption = "Red points indicate outliers\nDashed RED line represents the mean weight"
  ) +
  theme_classic() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40")
  )+
  stat_summary(fun = median, 
               geom = "point", 
               shape = 18,
               size = 2, 
               color = "gold")

# Print plot
print(weight_plot)

# Save high-quality plot
ggsave("Submession file/Fig_9.tiff", plot = weight_plot, width = 9, height = 6, dpi = 600, compression = "lzw")

# Statistical analysis of body weight vs Locality and Habitat
# 1. Assumption Checks --------------------------------------------------------
model <- aov(Weight_Kg ~ Habitat * Locality, data = final_Full_data3)

# Normality and homogeneity tests
shapiro_test <- shapiro.test(residuals(model))
levene_test <- leveneTest(Weight_Kg ~ Habitat * Locality, data = final_Full_data3)

# Quick assumption summary
cat("Assumption Check Results:",
    "\n- Normality (Shapiro-Wilk): p =", shapiro_test$p.value,
    "\n- Homogeneity (Levene's): p =", levene_test$`Pr(>F)`[1], "\n")

# 2. Statistical Analysis ----------------------------------------------------
final_Full_data$Habitat <- as.factor(final_Full_data3$Habitat)

if (shapiro_test$p.value >= 0.05 & levene_test$`Pr(>F)`[1] >= 0.05) {
  # Parametric approach
  anova_result <- summary(model)
  tukey_result <- TukeyHSD(model)
  effect_size <- etaSquared(model)
} else {
  # Non-parametric approach
  kw_habitat <- kruskal.test(Weight_Kg ~ Habitat, data = final_Full_data3)
  kw_locality <- kruskal.test(Weight_Kg ~ Locality, data = final_Full_data3)
  dunn_result <- dunnTest(Weight_Kg ~ Habitat, data = final_Full_data3, method = "bonferroni")
  
  # Corrected effect size calculation
  epsilon_habitat <- epsilonSquared(x = final_Full_data3$Weight_Kg,
                                    g = final_Full_data3$Habitat)
}
# 3. Results Reporting -------------------------------------------------------
sink("analysis_results.txt")

cat("STATISTICAL ANALYSIS RESULTS\n",
    "================================\n\n")

cat("KRUSKAL-WALLIS RESULTS\n",
    sprintf("\nHabitat: χ²(%d) = %.2f, p = %.3f", 
            kw_habitat$parameter, kw_habitat$statistic, kw_habitat$p.value),
    sprintf("\nLocality: χ²(%d) = %.2f, p = %.3f", 
            kw_locality$parameter, kw_locality$statistic, kw_locality$p.value),
    sprintf("\n\nEffect size (ε²): %.3f", epsilon_habitat),  # Removed $epsilon.sq
    "\n\nPOST-HOC TESTS (Dunn with Bonferroni)\n")
print(dunn_result)

sink()

# 4. Effect Size Calculation --------------------------------------------------
if(exists("dunn_result")) {
  # Simplified effect size calculation for significant pairs
  sig_pairs <- dunn_result$res %>% filter(P.adj < 0.05)
  if(nrow(sig_pairs) > 0) {
    cat("\nSIGNIFICANT PAIRWISE COMPARISONS\n")
    for(i in 1:nrow(sig_pairs)) {
      pair <- strsplit(sig_pairs$Comparison[i], " - ")[[1]]
      grp1 <- final_Full_data2$Weight_Kg[final_Full_data2$Habitat == pair[1]]
      grp2 <- final_Full_data2$Weight_Kg[final_Full_data2$Habitat == pair[2]]
      
      # Rank-biserial correlation
      r <- 1 - (2*wilcox.test(grp1, grp2)$statistic)/(length(grp1)*length(grp2))
      
      cat(sprintf("%s vs %s: Z = %.2f, p.adj = %.4f, r = %.2f\n",
                  pair[1], pair[2], 
                  sig_pairs$Z[i], sig_pairs$P.adj[i], r))
    }
  }
}
#####################
# 1. Enhanced Visualization ----------------------------------------------------
# Calculate the mean weight across all localities
mean_weight <- mean(final_Full_data2$Weight_Kg, na.rm = TRUE)
mean_weight
final_Full_data31<-final_Full_data3 %>% 
  filter(Weight_Kg < 300)
weight_plot1 <- ggplot(final_Full_data31, 
                       aes(x = reorder(Subfamily, Weight_Kg, median), 
                           y = Weight_Kg, 
                           fill = Habitat)) +
  geom_boxplot(width = 0.7, 
               alpha = 0.8,
               outlier.shape = 21,
               outlier.fill = "red",
               outlier.size = 2) +
  geom_jitter(width = 0.15, 
              alpha = 0.3, 
              size = 1.5) +
  geom_hline(yintercept = mean_weight, 
             linetype = "dashed", 
             color = "red", 
             size = 1) +  # Add dashed line at mean
  scale_fill_viridis_d(option = "D", 
                       begin = 0.2, 
                       end = 0.8) +
  labs(
    x = "Subfamily (Ordered by Median Weight)",
    y = "Estimated Weight (kg)",
    #caption = "Red points indicate outliers\nDashed blue line represents the mean weight"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.title = element_text(size = 14,face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1,size = 12),
        axis.text.y =  element_text(angle = 0, hjust = 1, vjust = 1,size = 12),
        panel.grid.major.x = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
        legend.text = element_text(size = 12)) +
  stat_summary(fun = median, 
               geom = "point", 
               shape = 18,
               size = 3, 
               color = "gold")

# Print plot
print(weight_plot1)

grouped_summary <- final_Full_data3 %>%
  group_by(Habitat, Subfamily) %>%
  summarise(
    Mean = mean(Weight_Kg, na.rm = TRUE),
    Median = median(Weight_Kg, na.rm = TRUE),
    SD = sd(Weight_Kg, na.rm = TRUE),
    Min = min(Weight_Kg, na.rm = TRUE),
    Max = max(Weight_Kg, na.rm = TRUE),
    IQR = IQR(Weight_Kg, na.rm = TRUE),
    N = sum(!is.na(Weight_Kg)),
    .groups = 'drop'
  )
grouped_summary
write.csv(grouped_summary,"Data/Final/taxa_wieghtmorphotypes.csv")
# Save high-quality plot
ggsave("Submession file/Fig_10.tiff", weight_plot1, width = 12, height = 8, dpi = 300)

# 2. Robust Statistical Analysis -----------------------------------------------

# Check for normality by locality (group-wise)
normality_tests <- final_Full_data2 %>%
  group_by(Subfamily) %>%
  shapiro_test(Weight_Kg) %>%
  mutate(normal = ifelse(p > 0.05, "Yes", "No"))

print(normality_tests)

# Perform appropriate test based on normality results
if(all(normality_tests$normal == "Yes")) {
  # Parametric tests
  cat("\n## Parametric Analysis ##\n")
  
  # Two-Way ANOVA (Habitat + Locality)
  anova_model <- aov(Weight_Kg ~ Habitat * Subfamily, 
                     data = final_Full_data2)
  
  # Check ANOVA assumptions
  par(mfrow = c(2,2))
  plot(anova_model)
  par(mfrow = c(1,1))
  
  # Robust ANOVA (type III SS)
  anova_results <- Anova(anova_model, type = "III")
  print(anova_results)
  
  # Post-hoc pairwise comparisons (Tukey HSD)
  if(any(anova_results$`Pr(>F)`[1:2] < 0.05, na.rm = TRUE)) {
    tukey_results <- TukeyHSD(anova_model)
    print(tukey_results)
    
    # Compact letter display
    tukey.cld <- multcompLetters4(anova_model, tukey_results)
    print(tukey.cld)
  }
  
} else {
  # Non-parametric tests
  cat("\n## Non-Parametric Analysis ##\n")
  
  # Kruskal-Wallis test
  kruskal_results <- kruskal.test(Weight_Kg ~ Subfamily, 
                                  data = final_Full_data2)
  print(kruskal_results)
  
  # Dunn's post-hoc test with BH correction
  dunn_results <- final_Full_data2 %>%
    dunn_test(Weight_Kg ~ Subfamily, 
              p.adjust.method = "BH") %>%
    add_xy_position(x = "Locality")
  
  print(dunn_results)
  
  # Add significance to plot
  weight_plot <- weight_plot + 
    stat_pvalue_manual(dunn_results, 
                       label = "p.adj.signif",
                       tip.length = 0.01,
                       step.increase = 0.1)
}

# 3. Effect Size Calculation ---------------------------------------------------
# Calculate eta-squared (effect size)
if(exists("anova_model")) {
  library(effectsize)
  eta_squared <- eta_squared(anova_model)
  print(eta_squared)
}

# 4. Final Output --------------------------------------------------------------
# Save statistical results
sink("weight_analysis_results_subfamily.txt")
cat("=== Normality Tests ===\n")
print(normality_tests)
cat("\n=== ANOVA Results ===\n")
print(anova_results)
if(exists("tukey_results")) {
  cat("\n=== Post-hoc Comparisons ===\n")
  print(tukey_results)
}
if(exists("eta_squared")) {
  cat("\n=== Effect Sizes ===\n")
  print(eta_squared)
}
sink()
################################
