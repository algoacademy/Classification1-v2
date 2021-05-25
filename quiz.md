# Classification 1 Quiz

This quiz is part of Algoritma Academy assessment process. Congratulations on completing the Classification in Machine Learning I! We will conduct an assessment quiz to test the practical classification model techniques that you have learned on the course. The quiz is expected to be taken in the classroom, please contact our team of instructors if you missed the chance to take it in class.

To complete this assignment, you are required to build your classification model to classify the characteristics of employees who have resigned and have not. Use Logistic Regression and k-Nearest Neighbor algorithms by following these steps:

# Data Exploration

Let us start by preparing and exploring the data first. In this quiz, you will be using the turnover of employee data (`turnover`). The data is stored as a .csv format in this repository as `turnover_balance.csv` file. Import your data using `read.csv` or `read_csv` and save as `turnover` object. Before building your classification model, you will need to perform an exploratory analysis to understand the data. Glimpse the structure of our `turnover` data! You can choose either `str()` or `glimpse()` function.

```
# your code here
```

Turnover data consists of 10 variables and 7.142 rows. This dataset is a human resource data that shows historical data of employee characteristics who will resign or not. Below is more information about the variable in the dataset:

  - `satisfaction_level`: the level of employee satisfaction working in a company
  - `last_evaluation`: employee satisfaction level at the last evaluation
  - `number_project`: the number of projects the employee has received
  - `average_monthly_hours`: average hours worked per month
  - `time_spend_company`: length of time in the company (years)
  - `work_accident`: presence or absence of work accident, 0 = none, 1 = there
  - `promotion_last_5years`: ever got a promotion in the last 5 years, 0 = no, 1 = yes
  - `division`: name of department or division
  - `salary`: income level, divided into low, medium and high
  - `left`: employee history data resigned, 0 = no, 1 = yes
  
In this quiz, we will try to predict whether or not the employee has a resignation tendency using the `left` column as our target variable. Please change the class of `Work_accident`, `left`, and `promotion_last_5years` column to be in factor class as it should be.

```
# your code here
```

For example, as an HR staff, we are instructed to investigate divisions that has a long history of an employee resigning and analyze their average monthly hours. Let's do some aggregation of `average_monthly_hours` for each division. Because you only focused on the employee who left, you should filter the historical data with the condition needed. 

Using **dplyr** functions, you can use `filter()`, then `group_by()` function by `division` and `summarise()` the mean of `average_montly_hours`, then arrange it based on `average_montly_hours` from high to low using `arrange()` function.

As an alternative, if you are more familiar using **base R** code style, you can filter the data using conditional subsetting `data["condition needed",]`, than assign it into `df_left` object. After that, you can aggregate `df_left` based on `division` and `average_montly_hours` column using `aggregate()` function. Don't forget to use `mean` in `FUN` parameter and assign it into `df_agg`. In order to get the ordered mean value from high to low of the `average_montly_hours`, you can use `order()` function in conditional subsetting `data[order(column_name, decreasing = T), ]`.

```
# your code here
```
___
1. Based on the aggregation data that you have analyzed, which are the top 3 divisions with the highest average of monthly hours?
  - [ ] Marketing, Accounting, Management
  - [ ] Accounting, Support, Sales
  - [ ] Technical, IT, Management
  - [ ] Technical, IT, Research and Development (RandD)
___

# Data Preprocessing

After conducting the data exploratory, we will go ahead and perform preprocessing steps before building the classification model. Before we build the model, let us take a look at the proportion of our target variable in the `left` column using `prop.table(table(data))` function.

```
# your code here
```

It seems like our target variable has a balance proportion between both classes. Before we build the model, we should split the dataset into train and test data in order to perform model validation. Split `turnover` dataset into 80% train and 20% test proportion using `sample()` function and use `set.seed()` with the seed 100. Store it as a `train` and `test` object.

> **Notes:** Make sure you use `RNGkind()` before splitting

```
RNGkind(sample.kind = "Rounding")
set.seed(100)
# your code here

```

Let's take a look at the distribution of our target variable in `train` data using `prop.table(table(data))` to make sure that the train data also have a balanced proportion of our target class. Please round the proportion to two decimal places using the `round()` function.

```
# your code here

```

___
2. Based on the result above, which statement below is most fitting?
  - [ ] The class distribution is not balanced, but it is not necessary to balance the class proportion.
  - [ ] The class distribution is balanced, but it is not necessary to balance the class proportion.
  - [ ] The class distribution is balanced, but we should also make sure that the test data set also have balanced proportion.
  - [ ] The class distribution is balanced, and it is important to balance the class proportion so that model can predict well in both classes.
___

# Logistic Regression Model Fitting

After we have split our dataset in train and test set, let's try to model our `left` variable using all of the predictor variables to build a logistic regression. Please use the `glm(formula, data, family = "binomial")` to do that and store your model under the `model_logistic` object. Remember, we are not using `turnover` dataset any longer, and we will be using `train` dataset instead.

```
# model_logistic <- glm()
```

Based on the `model_logictic` you have made above, take a look at the summary of your model using `summary()` function.

```
# your code here
```
___
3. Logistic regression is one of the interpretable models. We can explain how likely each variable predicts the class we observe. Based on the model summary above, what can be interpreted from the `salarymedium` coefficient?
  - [ ] The probability of an employee that received medium salary to resign is 1.50.
  - [ ] Employee who received medium salary is about 1.50 more likely to resign than the employee who received the other levels of salary.
  - [ ] Employee who received medium salary is about 4.48 more likely to resign than the employee who received high salary. 
___

# K-Nearest Neighbor Model Fitting

Now let's try to explore the classification model using the k-Nearest Neighbor algorithm. In the k-Nearest Neighbor algorithm, we need to perform one more step of data preprocessing. For both our `train` and `test` set, drop the categorical variable from each column except our `left` variable. Separate the predictor and target in-out `train` and `test` set.

```
# predictor variables in `train`
train_x <-

# predictor variables in `test`
test_x <-

# target variable in `train`
train_y <-

# target variable in `test`
test_y <-
```

Recall that the distance calculation for kNN is heavily dependent upon the measurement scale of the input features. If any variable that have high different range of value could potentially cause problems for our classifier, so let's apply normalization to rescale the features to a standard range of values.

To normalize the features in `train_x`, please using `scale()` function. Meanwhile, in testing set data, please normalize each features using the attribute *center* and *scale* of `train_x` set data.

Please look up to the following code as an example to normalize `test_x` data:

```
scale(data_test, center = attr(data_train, "scaled:center"),
scale = attr(data_train, "scaled: scale"))
```

Now it's your turn to try it in the code below:

```
# your code here

# scale train_x data
train_x <- scale()

# scale test_x data
test_x <- scale()
```

After we have done performing data normalizing, we need to find the right **K** to use for our K-NN model. In practice, choosing k depends on the difficulty of the concept to be learned and the
number of records in the training set data.

___
4. The method for getting K value, does not guarantee you to get the best result. But, there is one common practice for determining the number of K. What method can we use to choose the number of k?
  - [ ] use k = 5
  - [ ] number of row
  - [ ] square root by number of row 
___

After answering the questions above, please find the number of k in the following code:

Hint: If you got a decimal number, do not forget to round it.

```
# your code here

```
___
5. Which number should we use as `k`?
  - [ ] 85
  - [ ] 76
  - [ ] 75
  - [ ] 38

___

Using `k` value we have calculated in the section before, try to predict `test_y` using `train_x` dan `train_y` dataset. Please use the `knn()` function and store the result under the `model_knn` object. Use the following code to help you:

```
library(class)
model_knn <- knn(train = ______, test = ________, cl = _______, k = _____)
```

# Prediction

Now let's get back to our `model_logistic`. In this section, try to predict `test` data using `model_logistic` return the probability value using `predict()` function with `type = "response"` in the parameter function and store it under `prob_value` object.

```
prob_value <-
```

Because the prediction results in the logistic model are probabilities, we have to change them to categorical / class according to the target class we have. Now, given a threshold of 0.55, try to classify whether or not an employee can be predicted to resign. Please use `ifelse()` function and store the prediction result under the `pred_value` object.

```
pred_value <-
```


Based on the prediction value above, try to answer the following question.

___
6. In the prescriptive analytics stage, the prediction results from the model will be considered for business decision making. So, please take your time to check the prediction results. How many predictions do our `model_logistic` generate for each class?
  - [ ] class 0 = 614, class 1 = 815
  - [ ] class 0 = 717, class 1 = 712
  - [ ] class 0 = 524, class 1 = 905
 ___ 

# Model Evaluation

In the previous sections, we have performed a prediction using both Logistic Regression and K-NN algorithm. However, we need to validate whether or not our model did an excellent job of predicting unseen data. In this step, try to make the confusion matrix of model performance in the logistic regression model based on `test` data and `pred_value` and use the positive class is "1".

**Note:** do not forget to do the explicit coercion `as.factor()`.

```
# your code here
```

Make the same confusion matrix for `model_knn` prediction result of `test_y`.

```
# your code here
```

Let's say that we are working as an HR staff in a company and are utilizing this model to predict the probability of an employee resigning. As HR, we would want to know which employee is highly potential to resign so that we can take a precautionary approach as soon as possible. As a side note, the company is still recovering from a recent financial crisis so it is best to have most of the prediction correct to prevent overexploitation of budget. Now try to answer the following questions.

___
7. Which one is the right metric to evaluate the model performance based on the business case explained above?
  - [ ] Recall
  - [ ] Specificity  
  - [ ] Accuracy  
  - [ ] Precision  

___
8. Using the metrics of your answer in the previous question, which of the two models has a better performance in detecting resigning employees?
  - [ ] K-Nearest Neighbor  
  - [ ] Logistic Regression
  - [ ] Both have more or less similar performance

___
9.  Now, recall what we have learned about the advantage and limitation of each model. Which statement below is **NOT TRUE**?
  - [ ] Use kNN because it tends to have a higher performance than Logistic Regression and able to perform binary or multiclass classification.
  - [ ] Use Logistic regression because it is interpretable and can can process both numerical and categorical variables as predictor.
  - [ ] It is still better to use kNN than Logistic Regression to gain higher model performance, even when most of your predictor is categorical variables.
___

