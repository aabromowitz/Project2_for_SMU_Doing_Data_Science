# SMU_DDS_Project2
A repository with files related to Project #2 in the Doing Data Science course at SMU

Purpose: <br>
We were given a dataset of client data from the fake company DDS Analytics.  The goal was to perform an EDA of that data, then create models to predict Attrition (Yes/No categorical variable) and Monthly Income (numeric variable).  We also needed to create a PPT, give a 
7-8 minute presentation, create an RShiny App, create an RMarkdown file, generate predictions for provided datasets with missing Attrition and Monthly Income values, and create a webpage with all of this.

Webpage: https://aabromowitz.github.io/projects/CaseStudy2/

Explanation of predictions: <br>
I made 2 predictions for each Attrition and Monthly Income.  Attrition_Model_1.xlsx uses a KNN model with 6 variables, Attrition_Model_2.xlsx uses a Naive Bayes model with 50 variables, MonthlyIncome_Model_1 uses a linear regression model with 10 variables, and MonthlyIncome_Model_2 uses a linear regression model with 55 variables.  I was worried that the models with 50+ variables would have a potential for over-fit, so I made Models #1 my "official" predictions.  But I was curious how the models with a large number of variables would do, so I included them.
Attrition_Model_1.xlsx only has 3 "Yes" values, which doesn't seem right.  However, I didn't want my final predictions to change based on that.  So I'm still using Attrition_Model_1.xlsx as my "official" predictions, even though there's a high chance they aren't very good.  In general, Attrition Model 1 predicted less Yes values, but those Yes values were more likely to be accurate.  On the flip side, this model only got above .6 Sensitivity 80% of the time, so there was definitely a chance that it didn't pick enough correct Yes values.

Contents of repository:<br>
PPT slides for the presentation <br>
4 Excel files of the predictions (2 for Attrition, 2 for Monthly Income) <br>
RMarkdown File <br>
R Scripts that I used for my investigation <br>
