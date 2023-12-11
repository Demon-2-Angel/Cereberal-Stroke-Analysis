# Cereberal-Stroke-Analysis
<p align="center">
<img src="https://github.com/Demon-2-Angel/Cereberal-Stroke-Analysis/blob/main/Images/Brain%20Stroke.png">
</p>
# Followed Process

## Read Data:

The script starts by importing necessary libraries (`pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`) and reading a CSV file into a DataFrame (`df`).

## Exploratory Data Analysis (EDA):

Basic exploration of the dataset using `head()`, `describe()`, and checking for missing values using `isnull().sum()`.

## Handling Categorical Variables:

One-hot encoding is performed on categorical variables using `pd.get_dummies()`.

## Handling Missing Values:

Missing values are imputed using the k-nearest neighbors algorithm (`KNNImputer` from `sklearn.impute`).


## Feature Scaling and Train-Test Split:

Features are scaled using `MinMaxScaler`, and the dataset is split into training and testing sets.

## Model Selection:

Several classification models are chosen (`KNeighborsClassifier`, `GaussianNB`, `DecisionTreeClassifier`, and `RandomForestClassifier`) for initial testing.


## Model Evaluation Without Resampling:

Classification reports are generated for each model to evaluate their performance on the imbalanced dataset.

<p align="center">
<img src="https://github.com/Demon-2-Angel/Cereberal-Stroke-Analysis/blob/main/Images/Before%20Sampling.png">
</p>


## OverSampling (SMOTE):

The script uses the Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class.



## Model Evaluation After OverSampling:

The same models are re-trained and evaluated on the oversampled dataset.

<p align="center">
<img src="https://github.com/Demon-2-Angel/Cereberal-Stroke-Analysis/blob/main/Images/OverSampling.png">
</p>


## UnderSampling:

Random under-sampling is performed to balance the class distribution.

## Model Evaluation After UnderSampling:

The models are re-trained and evaluated on the undersampled dataset.

<p align="center">
<img src="https://github.com/Demon-2-Angel/Cereberal-Stroke-Analysis/blob/main/Images/UnderSampling.png">
</p>


## Combining OverSampling and UnderSampling (SMOTEENN):

The SMOTEENN technique, which combines SMOTE and Edited Nearest Neighbours (ENN), is applied.

## Model Evaluation After Combining OverSampling and UnderSampling:

The models are re-trained and evaluated on the combined dataset.

<p align="center">
<img src="https://github.com/Demon-2-Angel/Cereberal-Stroke-Analysis/blob/main/Images/After%20Over%20%26%20Under%20Sampling.png">
</p>


## Conclusion:

- The script provides classification reports for each model after different resampling techniques.
- It highlights that resampling techniques, particularly SMOTEENN, improve the model's ability to identify cases positive for stroke.
