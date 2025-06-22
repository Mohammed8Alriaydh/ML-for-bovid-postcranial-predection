# ML-for-bovid-postcranial-predection

This project focuses on classifying bovidae subfamilies and predicting their habitats using morphological data from astragali and proximal phalanges. It includes handling missing data, training machine learning models, evaluating performance, conducting PCA for visualization, and statistical analysis. The workflow also predicts habitats and body weights for fossil specimens.

**Dependencies**
Ensure the following R packages are installed:

install.packages(c("tidyverse", "caret", "randomForest", "e1071", "MASS", "Amelia", "VIM", "pROC", "ggpubr", "FactoMineR", "factoextra", "ggforce", "rstatix", "car", "FSA", "rcompanion", "officer"))

**Data Preparation**
**Datasets**
bovidae_training_Missing_data.csv: Astragali measurements with missing values.
bovidae_training_data_random_forest.csv: Habitat labels for training.
Ast_sub_data.csv, pxp_sub_data.csv: Metadata for fossil specimens.
Missing Data Handling: Uses multiple imputation (Amelia) to address missing values.

**Model Training and Evaluation**
Models: Random Forest, Linear Discriminant Analysis (LDA), Support Vector Machine (SVM).
Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
Cross-Validation: Data split into training/test sets (80/20).

**Prediction**
Unknown Data: Classifies subfamilies and habitats for fossil specimens.
Weight Estimation: Uses regression equations from literature to estimate body weight.

**Visualization and Statistical Analysis**
PCA: Reduces dimensionality and visualizes morphological space.
Non-parametric Tests: Kruskal-Wallis and Dunn's tests for group comparisons.
Plots: Includes boxplots, bar charts, and PCA biplots.
