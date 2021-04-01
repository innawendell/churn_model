## StreamLabs Telecom Churn
**1. Briefly describe your predictive churn model. How did you select the variables to be included in the model?**

We experimented with three models, including Logistic Regression (with l2 and l1 regularization), and RandomForest, and CatBoost.
Due to time constraints, we opted for embedded methods of feature selection (l1 and l2 regularization in case of Logistic Regression that lower coefficients for less relevant features) and filter methods of feature selection. We used the chi-square test to filter out the features that with a low impact on churn (p-value below 0.05) for categorical features, . For numerical features, we used feature correlations and filtered out such features that the absolute value of the correlation coefficient below 0.01, which was based on observed correlation coefficients. This threshold can be experimented with in the future. Since we identified class imbalance (the number of non-churning customers was almost double the number of churning customers). We addressed that issue by undersampling the dominant class (non-churning customers). We also experimented with oversampling and generating synthetic training data using smote, but undersampling gave better performance. Due to class imbalance, we realized that it would be inappropriate to use accuracy as evaluation metric. A model would get a high accuracy by simply selecting non-churning customers more often since it is the dominant class. Therefore, we used the roc-auc score for model evaluation since it allows us to evaluate how many true churning customers (true positives) we get vs. falsely identified churning customers (false positives). We used 3-fold cross-validation to fine-tuned hyperparameters. The best hyperparameters for the logistic regression were `alpha` (C) = 1000 and `regularization` (penalty) = l2. The best hyperparameters for the Random Forest were `max_depth` = 12 and `n_estimators` = 700. For the CatBoost Model, the best hyperparameters were `max_depth` = 4, `iterations` = 300, and `learning_rate` = 0.1. During the initial stages, we experimented with XGBoost, but CatBoost seemed to outperform XGBoost consistently. The best performing model (CatBoost) gave us roc-auc on the cross-validation on the training set of 0.6615 and on the test set of 0.673. There is much room for improvement. It would be ideal to get a model that could get roc-auc score above 0.7.

**2. What are the key factors that predict customer churn, based on your findings? Do these factors make sense? What is their relative importance?**

The analysis of the Shap values for the `Logistic Regression` revealed the following features are associated with churn:
- Low (blue in color) average monthly minutes;
- High number of days (red in color) on current phone;
- Low number (blue) of calls during peak hours;
- High (red) overage minutes used;
- High number (red) of minutes received;
- Low number of active subscribers;
- High average number of dropped voice calls;
- High number of unique users;
- Fewer instances of having the highest credit score of AA;
- Higher handset price;
- Lower number of of unique phone models;
- Lower average number of inbound calls;
- Higher number of refurbished phones;
- Lower age of the first household member.
- Higher average monthly revenue;
- Higher average number of unanswered calls;
- Higher number of retention calls;
- Negative % of change in minutes of use;
- Mostly lower average number of calls during off-peak hours.
- Fewer instances of having the highest credit score of A.

The analysis of the Shap values for the `CatBoost` showed that the following features are associated with churn:
- High number of days (red in color) on current phone;
- Low (mostly blue, mixed color means interaction with another feature) average monthly minutes used;
- Fewer (mostly blue) months in service;
- Negative (blue) % of change in the minutes used;
- High (red) overage minutes used;
- Predominantly lower recurring charge;
- Mostly lower age of the first household member;
- Fewer instances of having the highest credit score of AA;
- Higher number of dropped calls;
- Lower average number of inbound calls;
- More calls to retention;
- Higher number of refurbished phones;
- Higher number of unique subscribers;
- Lower (mostly) average number of calls to customer care;
- Higher handset price;
- Higher average number of blocked voice calls;
- Higher number of roaming calls;
- Higher number of minutes received;
- Higher number of unanswered phone calls;
- Not having web capabilities.

Obtaining Shap values for Random Forest takes a long time, which was impossible to spend at this moment. Instead, we obtained feature importances from the model itself. Unfortunately, these importances do not tell us whether a low or high feature value is associated with churn. We should perform more investigation with Shap values in the future.

Top Ten Features Impacting Churn:

- Number of days of current phone;
- Months in service;
-  % change in minutes of use;
-  Average monthly minutes of use;
-  Total recurring charge;
-  Average monthly revenue;
-  Overage minutes used;
-  Average number of calls during peak hours;
-  Average number of calls during off-peak hours;
-  Minutes of voice calls received.

**3. What would you do to improve your work/get greater results?**

If we had more time, we could do more data engineering and create more features. Additionally, we could use wrapper feature selection methods (e.g., with the forward selection or backward elimination), if we had more time. We could also obtain more training data, depending on availability. Finally, we could have fine-tuned more hyperparameters for the Random Forest. Our models performed better than random, yet, there is much room for improvement. It would be useful to get the models to get roc-auc of at least 0.7. It is also useful to define our business goals (whether we care more about true positives (identifying as many churning customers as possible) or false positives (worry to give incentives to the customers who we think are churning but actually are not). This decision will impact the threshold we use to make predictions and determine the confusion matrix we get as a result. We have obtained shap values and feature importances, which indicated the features impacting churn. However, since our models did not perform extremely well, these importances should be taken with a grain of salt. The analysis should be repeated once we get a better roc-auc score.

**4. How long did it take you to complete this assignment?**

The assignment took 15 hours (over the course of 3 days).

**5. What about this assignment did you find most challenging?**

There were issues with the data: missing values and negative values where not supposed to be. None of the features had a very high correlation with churn. Tackling categorical and numerical features required different approaches. Understanding some of the factors impacting certain features (e.g., the number of users on the account) required analysis.

**6. What about this assignment did you find unclear?**

The timeline to complete the assignment was not exactly clear: the work quality would have been better if there was no pressure to complete the assignment fast.

**7. Do you feel like this assignment was an appropriate level of difficulty?**

Yes, I believe it is a good idea to ask a data scientist candidate to create a predictive model similar to the model they will be working on in real life. I wish only that all candidates were given the same amount of time to complete thetsk.


### Notebooks
There are two notebooks for this assignment:
1. `Churn_Model.ipynb`: contains data exploration and model training. Since training and test split happens randomly, one may arrive at slightly different results if one is to rerun the notebook.
2. `Validation_on_Test_Set.ipynb`: contains the pipeline to evaluate the model on the test set (due to numerous feature transformations) as well as the evaluation itself. It also has feature imporances based on Shap Values.

### Models
The `models` folder has the pre-trained supervised models as well as the scaler. These models are available in the zip but no on GitHub due to their size.


### Helper Functions
The python file `helper_functions.py` contains the functions we used for feature transformations. They were put in a separate file so that they can be shared by different notebooks as well as to improve readability.

