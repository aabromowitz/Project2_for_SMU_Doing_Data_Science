###########Beginning Logic###############################

# Packages
library(tidyverse)
# install.packages("RCurl")
library(RCurl)
library(e1071)
library(caret)
library(mltools)
library(data3.table)
library(xgboost)
library(openxlsx)

# Pull in data3
Sys.setenv("AWS_ACCESS_KEY_ID" = "AKIARXUIWXWHQS6F23XS",
           "AWS_SECRET_ACCESS_KEY" = "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")
data3 <- read.table(textConnection(getURL(
  "https://s3.us-east-2.amazonaws.com/ddsproject1/CaseStudy2-data3.csv"
)), sep=",", header=TRUE)

# Reset some of the numeric looking variables to words.
data3$Education[data3$Education == 1] <- 'Below College'
data3$Education[data3$Education == 2] <- 'College'
data3$Education[data3$Education == 3] <- 'Bachelor'
data3$Education[data3$Education == 4] <- 'Master'
data3$Education[data3$Education == 5] <- 'Doctor'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 1] <- 'Low'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 2] <- 'Medium'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 3] <- 'High'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 4] <- 'Very High'
data3$JobInvolvement[data3$JobInvolvement == 1] <- 'Low'
data3$JobInvolvement[data3$JobInvolvement == 2] <- 'Medium'
data3$JobInvolvement[data3$JobInvolvement == 3] <- 'High'
data3$JobInvolvement[data3$JobInvolvement == 4] <- 'Very High'
data3$JobSatisfaction[data3$JobSatisfaction == 1] <- 'Low'
data3$JobSatisfaction[data3$JobSatisfaction == 2] <- 'Medium'
data3$JobSatisfaction[data3$JobSatisfaction == 3] <- 'High'
data3$JobSatisfaction[data3$JobSatisfaction == 4] <- 'Very High'
data3$PerformanceRating[data3$PerformanceRating == 1] <- 'Low'
data3$PerformanceRating[data3$PerformanceRating == 2] <- 'Good'
data3$PerformanceRating[data3$PerformanceRating == 3] <- 'Excellent'
data3$PerformanceRating[data3$PerformanceRating == 4] <- 'Outstanding'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 1] <- 'Low'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 2] <- 'Medium'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 3] <- 'High'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 4] <- 'Very High'
data3$WorkLifeBalance[data3$WorkLifeBalance == 1] <- 'Bad'
data3$WorkLifeBalance[data3$WorkLifeBalance == 2] <- 'Good'
data3$WorkLifeBalance[data3$WorkLifeBalance == 3] <- 'Better'
data3$WorkLifeBalance[data3$WorkLifeBalance == 4] <- 'Best'

# Set variables to factors
data3_original <- data3
data3$Attrition <- factor(data3$Attrition,levels=c("Yes","No")) # Making "Yes" the true value for bayes
data3$BusinessTravel <- as.factor(data3$BusinessTravel)
data3$Department <- as.factor(data3$Department)
data3$EducationField <- as.factor(data3$EducationField)
data3$Gender <- as.factor(data3$Gender)
data3$JobRole <- as.factor(data3$JobRole)
data3$MaritalStatus <- as.factor(data3$MaritalStatus)
data3$OverTime <- as.factor(data3$OverTime)
data3$Education <- factor(data3$Education,levels=c("Below College","College","Bachelor","Master","Doctor"))
data3$EnvironmentSatisfaction <- factor(data3$EnvironmentSatisfaction,levels=c("Low","Medium","High","Very High"))
data3$JobInvolvement <- factor(data3$JobInvolvement,levels=c("Low","Medium","High","Very High"))
data3$JobLevel <- as.factor(data3$JobLevel)
data3$JobSatisfaction <- factor(data3$JobSatisfaction,levels=c("Low","Medium","High","Very High"))
data3$PerformanceRating <- factor(data3$PerformanceRating,levels=c("Low","Good","Excellent","Outstanding"))
data3$RelationshipSatisfaction <- factor(data3$RelationshipSatisfaction,levels=c("Low","Medium","High","Very High"))
data3$StockOptionLevel <- as.factor(data3$StockOptionLevel)
data3$WorkLifeBalance <- factor(data3$WorkLifeBalance,levels=c("Bad","Good","Better","Best"))

# Make a YearsAtCompany_Capped variable and cap it at 30
data3$YearsAtCompany_Capped <- pmin(data3$YearsAtCompany,30)

# Create lists of the variables
# List all the categorical and numeric variables
cat_vars <- c('Attrition','BusinessTravel','Department','Education','EducationField',
              'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole',
              'JobSatisfaction','MaritalStatus','OverTime','PerformanceRating',
              'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance')
num_vars <- c('Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyRate',
              'NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
              'TrainingTimesLastYear', 'YearsAtCompany','YearsInCurrentRole',
              'YearsSinceLastPromotion','YearsWithCurrManager', 'YearsAtCompany_Capped')

# Create square vars
nvars <- length(num_vars)
squared_vars <- c()
data3_no_squared <- data3
for (ii in 1:nvars){
  var <- num_vars[ii]
  var_squared <- paste(var,"_squared",sep="")
  data3[,var_squared]<-data3[,var]*data3[,var]
  squared_vars <- c(squared_vars,var_squared)
}

# Get onehot encoded variables
before_onehot <- data3
data3 <- one_hot(as.data3.table(data3))
data3 <- as.data3.frame(data3)
nvar <- length(cat_vars)
for (ii in 1:nvar){
  var <- cat_vars[ii]
  data3[,var] <- before_onehot[,var]
}
names(data3) <- gsub(' ','_',names(data3))
names(data3) <- gsub('&','AND',names(data3))
names(data3) <- gsub('-','_',names(data3))
col_names <- names(data3)
onehot_vars <- col_names[grepl('_', names(data3))]
onehot_vars <- onehot_vars[!grepl('squared', onehot_vars)]

# Make cross products of variables
data3_before_cross <- data3
to_cross_vars <- c(num_vars,onehot_vars)
nvars <- length(to_cross_vars)
cross_vars <- c()
for (ii in 1:nvars){
  for(jj in 1:nvars){
    if (ii<jj){
      var1 <- to_cross_vars[ii]
      var2 <- to_cross_vars[jj]
      new_col <- data3[,var1]*data3[,var2]
      if(sum(new_col)!=0){
        new_var <- paste(var1,var2,sep="X")
        data3[,new_var]<-new_col
        cross_vars <- c(cross_vars,new_var)
      }
    }
  }
}

# Combine all the variables
all_vars <- c(cat_vars, num_vars, squared_vars, onehot_vars, cross_vars)

#########################################################

# Try scaling the data3 first, maybe Z scaling (but not to Monthly Salary)

# Look at the p-values for original vars

# If you notice any data3 that's like 5 standard deviations away or more, maybe examine it

# Try running the function
seed <- 1
prop_test <- .1
thresh <- 10 # This becomes 900 in the code, to match 3,000 for 300 results
num_tests <- 200
num_no_change <- 50
num_models <- 10
# start_formula <- 'MonthlyIncome~JobLevel+TotalWorkingYears'
start_formula <- ''
# start_num <- 3
start_num <- 1
vars_df <- add_vars_reg(data3, all_vars, seed, prop_test, thresh, num_tests, num_no_change, num_models, start_formula, start_num)
write.xlsx(vars_df,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing data3 Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/vars_df.xlsx',colNames = TRUE)

# Saving off var_df
var_df <- data3.frame(Variables=all_vars)
var_df$p <- 0
nvar <- length(all_vars)
for(ii in 1:nvar){
  var<-all_vars[ii]
  # print(var)
  test <- cor.test(unclass(data3[,'MonthlyIncome']),unclass(data3[,var]))
  var_df$p[ii] <- test$p.value
}
var_df$p[is.na(var_df$p)] <- 1
write.xlsx(var_df,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing data3 Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/var_df.xlsx',colNames = TRUE)


# Debugging
formula_str <- 'MonthlyIncome ~ JobLevel'
formula_obj <- as.formula(formula_str)
model <- lm(formula_obj, data3 = data3)

# Trying to comine models
fit1 <- lm(Sepal.Length ~ Sepal.Width, data3 = iris)
fit2 <- lm(Petal.Length ~ Petal.Width, data3 = iris)
fits <- list(fit1, fit2)
class(fits) <- "lmList"
library(nlme)
confint(fits)
preds<-predict(fits,data3)

# That didn't work very well.  
# I wonder what some variables that are associated with MonthlyIncome are.
# It looks like the following are highly correlated with MonthlyIncome:
# JobLevel, TotalWorkingYears, YearsAtCompany_Capped
formula_str <- 'MonthlyIncome ~ JobLevel'
formula_obj <- as.formula(formula_str)
model <- lm(formula_obj, data3 = data3)
preds <- predict(model,data3)
head(preds)
head(as.numeric(data3[,"MonthlyIncome"]))
err <- sqrt(sum((preds - as.numeric(data3[,"MonthlyIncome"]))^2)/nrow(data3)) # 1251

# Oh, that's way easier to get
num_passed <- 0
metric <- 0
thresh <- 3000
# formula_str <- 'MonthlyIncome ~  JobLevel+TotalWorkingYears+AgeXTotalWorkingYears+TotalWorkingYears_squared+JobRole_Manager+TotalWorkingYearsXYearsAtCompany_Capped+TotalWorkingYearsXBusinessTravel_Travel_Rarely+JobLevel_1XPerformanceRating_Excellent+Attrition_NoXJobLevel_4+AgeXJobRole_Research_Director+TotalWorkingYearsXJobRole_Research_Director+JobRole_Research_Director+MonthlyRateXJobRole_Manager+YearsAtCompany_Capped_squared+Attrition_NoXJobRole_Research_Director+YearsAtCompany_CappedXJobRole_Manager+YearsAtCompanyXYearsAtCompany_Capped+YearsAtCompany_CappedXJobLevel_4+MonthlyRateXYearsAtCompany+YearsSinceLastPromotionXYearsAtCompany_Capped+Education_BachelorXJobLevel_1+JobRole_Laboratory_Technician+AgeXJobLevel_3+AgeXJobRole_Research_Scientist+JobLevel_1XJobRole_Laboratory_Technician+TrainingTimesLastYearXJobRole_Research_Scientist+HourlyRateXJobRole_Laboratory_Technician+AgeXJobRole_Laboratory_Technician+MonthlyRateXYearsInCurrentRole+JobLevel_1XRelationshipSatisfaction_High+TrainingTimesLastYearXJobLevel_3+TotalWorkingYearsXEducationField_Medical+DailyRateXJobLevel_3+Attrition_YesXJobLevel_1+DailyRateXYearsInCurrentRole+DistanceFromHomeXYearsAtCompany_Capped+TotalWorkingYearsXJobRole_Research_Scientist+TotalWorkingYearsXJobRole_Laboratory_Technician+YearsInCurrentRoleXDepartment_Research_AND_Development+YearsInCurrentRoleXJobRole_Research_Scientist+YearsSinceLastPromotionXDepartment_Research_AND_Development+YearsAtCompanyXJobRole_Laboratory_Technician+JobRole_Research_ScientistXOverTime_Yes+Education_BachelorXJobLevel_3+YearsInCurrentRoleXGender_Male+YearsInCurrentRoleXJobRole_Laboratory_Technician+DistanceFromHomeXYearsWithCurrManager+DistanceFromHomeXJobRole_Sales_Representative+YearsSinceLastPromotionXJobRole_Research_Scientist+DistanceFromHomeXYearsSinceLastPromotion+YearsWithCurrManagerXJobSatisfaction_Medium+YearsSinceLastPromotionXJobRole_Laboratory_Technician+YearsSinceLastPromotionXRelationshipSatisfaction_Medium+TotalWorkingYearsXJobRole_Sales_Representative+TotalWorkingYearsXJobLevel_2+MonthlyRateXJobLevel_2+YearsSinceLastPromotionXJobRole_Sales_Executive'
formula_str <- 'MonthlyIncome ~  JobLevel'
formula_obj <- as.formula(formula_str)
seed <- 5
set.seed(seed)
for (jj in 1:num_tests){
  indices = sample(seq(1:nrow(data3)),round(prop_test*nrow(data3)))
  validation = data3[indices,]
  train = data3[-indices,]
  
  # Average of serveral models seems hard, so I'll just use one model
  model <- lm(formula_obj, data3 = train)
  
  # Test model
  pred <- predict(model, validation)
  err <- sqrt(sum((pred-as.numeric(validation[,"MonthlyIncome"]))^2)/nrow(validation))
  if(err<thresh) {
    num_passed <- num_passed + 1
  }
  metric <- metric + err
  
}
num_passed
metric / num_tests # first 3 variables is 1,254, crazy formula is 968

# Plot Monthly Income vs Job Level
data3 %>% ggplot(aes(x=MonthlyIncome,fill=JobLevel)) +
  geom_histogram() + xlab('Monthly Income') + ylab('Count') + 
  ggtitle('Monthly Income vs Job Level') +
  scale_fill_manual("Attrition", 
                    values = c("1" = "#13294b", "2" = "darkblue", "3" = "blue", 
                               "4" = "cornflowerblue", "5" = "lightblue")) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 7), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Monthly Income vs Education
data3 %>% ggplot(aes(x=MonthlyIncome)) +
  geom_histogram(fill="#13294b") + xlab('Monthly Income') + ylab('Count') + 
  ggtitle('Monthly Income vs Education Level') +
  facet_wrap(~ Education, ncol=1, scales='free_y') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 7), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Monthly Income vs Total Working Years
data3 %>% ggplot(aes(x=TotalWorkingYears, y=MonthlyIncome)) +
  geom_point(color="#13294b") + xlab('Total Working Years') + ylab('Monthly Income') + 
  ggtitle('Monthly Income vs Total Working Years') +
  # geom_smooth(method = "lm", formula = y ~ x) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 10), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Monthly Income vs Years Since Last Promotion
data3 %>% ggplot(aes(x=YearsSinceLastPromotion, y=MonthlyIncome)) +
  geom_point(color="#13294b") + xlab('Years Since Last Promotion') + ylab('Monthly Income') + 
  ggtitle('Monthly Income vs Years Since Last Promotion') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 10), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot MonthlyIncome vs AgeXTotalWorkingYears
data3 %>% ggplot(aes(x=AgeXTotalWorkingYears, y=MonthlyIncome)) +
  geom_point(color="#13294b") + xlab('Age X Total Working Years') + ylab('Monthly Income') + 
  ggtitle('Monthly Income by AgeXTotalWorkingYears') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 10), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot MonthlyIncome vs TotalWorkingYears_squared
data3 %>% ggplot(aes(x=TotalWorkingYears_squared, y=MonthlyIncome)) +
  geom_point(color="#13294b") + xlab('Total Working Years Squared') + ylab('Monthly Income') + 
  ggtitle('Monthly Income by Total Working Years Squared') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 10), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Determine how well 3 variables works for Linear Regression
seed <- 20
prop_test <- .1
num_tests <- 200
#formula_str <- 'MonthlyIncome ~  JobLevel'
# formula_str <- 'MonthlyIncome ~  JobLevel+TotalWorkingYears+AgeXTotalWorkingYears+TotalWorkingYears_squared+JobRole_Manager+TotalWorkingYearsXYearsAtCompany_Capped+TotalWorkingYearsXBusinessTravel_Travel_Rarely+JobLevel_1XPerformanceRating_Excellent+Attrition_NoXJobLevel_4+AgeXJobRole_Research_Director+TotalWorkingYearsXJobRole_Research_Director+JobRole_Research_Director+MonthlyRateXJobRole_Manager+YearsAtCompany_Capped_squared+Attrition_NoXJobRole_Research_Director+YearsAtCompany_CappedXJobRole_Manager+YearsAtCompanyXYearsAtCompany_Capped+YearsAtCompany_CappedXJobLevel_4+MonthlyRateXYearsAtCompany+YearsSinceLastPromotionXYearsAtCompany_Capped+Education_BachelorXJobLevel_1+JobRole_Laboratory_Technician+AgeXJobLevel_3+AgeXJobRole_Research_Scientist+JobLevel_1XJobRole_Laboratory_Technician+TrainingTimesLastYearXJobRole_Research_Scientist+HourlyRateXJobRole_Laboratory_Technician+AgeXJobRole_Laboratory_Technician+MonthlyRateXYearsInCurrentRole+JobLevel_1XRelationshipSatisfaction_High+TrainingTimesLastYearXJobLevel_3+TotalWorkingYearsXEducationField_Medical+DailyRateXJobLevel_3+Attrition_YesXJobLevel_1+DailyRateXYearsInCurrentRole+DistanceFromHomeXYearsAtCompany_Capped+TotalWorkingYearsXJobRole_Research_Scientist+TotalWorkingYearsXJobRole_Laboratory_Technician+YearsInCurrentRoleXDepartment_Research_AND_Development+YearsInCurrentRoleXJobRole_Research_Scientist+YearsSinceLastPromotionXDepartment_Research_AND_Development+YearsAtCompanyXJobRole_Laboratory_Technician+JobRole_Research_ScientistXOverTime_Yes+Education_BachelorXJobLevel_3+YearsInCurrentRoleXGender_Male+YearsInCurrentRoleXJobRole_Laboratory_Technician+DistanceFromHomeXYearsWithCurrManager+DistanceFromHomeXJobRole_Sales_Representative+YearsSinceLastPromotionXJobRole_Research_Scientist+DistanceFromHomeXYearsSinceLastPromotion+YearsWithCurrManagerXJobSatisfaction_Medium+YearsSinceLastPromotionXJobRole_Laboratory_Technician+YearsSinceLastPromotionXRelationshipSatisfaction_Medium+TotalWorkingYearsXJobRole_Sales_Representative+TotalWorkingYearsXJobLevel_2+MonthlyRateXJobLevel_2+YearsSinceLastPromotionXJobRole_Sales_Executive'
formula_str <- 'MonthlyIncome ~ JobLevel+TotalWorkingYears+TotalWorkingYears_squared+JobRole_Manager+TotalWorkingYearsXYearsAtCompany_Capped+TotalWorkingYearsXBusinessTravel_Travel_Rarely+Attrition_NoXJobLevel_4+AgeXJobRole_Research_Director+JobRole_Research_Director+YearsAtCompany_CappedXJobRole_Manager'
thresh <- 3000

metric <- 0
num_passed <- 0
set.seed(seed) # re-setting the seed for some sort of predictability
for (jj in 1:num_tests){
  indices = sample(seq(1:nrow(data3)),round(prop_test*nrow(data3)))
  validation = data3[indices,]
  train = data3[-indices,]
  model <- lm(as.formula(formula_str), data3 = train)
  pred <- predict(model, validation)
  rmse <- sqrt(sum((pred-as.numeric(validation[,"MonthlyIncome"]))^2)/nrow(validation))
  if(rmse<thresh){
    num_passed <- num_passed+1
  }
  metric <- metric + rmse
}
metric / num_tests
num_passed

# What about XGBoost for the linear model?
seed <- 5
prop_test <- .1
set.seed(seed)
indices = sample(seq(1:nrow(data3)),round(prop_test*nrow(data3)))
validation = data3[indices,]
train = data3[-indices,]

vals <- c('TotalWorkingYears','AgeXTotalWorkingYears','TotalWorkingYears_squared','AgeXJobRole_Research_Director')
target <- train$MonthlyIncome
feature_data3 <- train[,vals]
feature_matrix <- as.matrix(feature_data3)
dtrain <- xgb.DMatrix(data3 = feature_matrix, label = target)
bstDMatrix <- xgboost(data3 = dtrain, nthread = 2, nrounds = 10, objective = "reg:squarederror")
validation_matrix <- as.matrix(validation[,vals])
validation_dmatrix <- xgb.DMatrix(data3 = validation_matrix)
pred <- predict(bstDMatrix, validation_dmatrix)
rmse <- sqrt(sum((pred-as.numeric(validation[,'MonthlyIncome']))^2)/nrow(validation)) # It seems like it's doing well enough to give it a try using the approach I used before
rmse 
# It seems like it does a bad job of handling categorical data3.  
# This makes me skeptical that it will be able to beat lm, 
#   since JobLevel is such a good predictor for MonthlyIncome

# Try running the function using the cutoff
seed <- 1
prop_test <- .1
thresh <- 3000
num_tests <- 200
num_no_change <- 50
start_formula <- ''
start_num <- 1
txt_file <- 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing data3 Science/Units/Unit 14 and 15 - Project 2 (Individual)/Txt_Files/Save_Lin.txt'
start_best <- 1000000000
amount_to_improve <- 1
var_to_predict <- 'MonthlyIncome'
vars_df <- add_vars_reg(data3, all_vars, seed, prop_test, thresh, num_tests, 
                        num_no_change, start_formula, start_num, txt_file, 
                        start_best, amount_to_improve, var_to_predict)
write.xlsx(vars_df,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing data3 Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/vars_df.xlsx',colNames = TRUE)

# Prepping data3 for final predictions
data3 <- read.table(textConnection(getURL(
  "https://s3.us-east-2.amazonaws.com/ddsproject1/CaseStudy2CompSet%20No%20Salary.csv"
)), sep=",", header=TRUE)

# Reset some of the numeric looking variables to words.
data3$Education[data3$Education == 1] <- 'Below College'
data3$Education[data3$Education == 2] <- 'College'
data3$Education[data3$Education == 3] <- 'Bachelor'
data3$Education[data3$Education == 4] <- 'Master'
data3$Education[data3$Education == 5] <- 'Doctor'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 1] <- 'Low'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 2] <- 'Medium'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 3] <- 'High'
data3$EnvironmentSatisfaction[data3$EnvironmentSatisfaction == 4] <- 'Very High'
data3$JobInvolvement[data3$JobInvolvement == 1] <- 'Low'
data3$JobInvolvement[data3$JobInvolvement == 2] <- 'Medium'
data3$JobInvolvement[data3$JobInvolvement == 3] <- 'High'
data3$JobInvolvement[data3$JobInvolvement == 4] <- 'Very High'
data3$JobSatisfaction[data3$JobSatisfaction == 1] <- 'Low'
data3$JobSatisfaction[data3$JobSatisfaction == 2] <- 'Medium'
data3$JobSatisfaction[data3$JobSatisfaction == 3] <- 'High'
data3$JobSatisfaction[data3$JobSatisfaction == 4] <- 'Very High'
data3$PerformanceRating[data3$PerformanceRating == 1] <- 'Low'
data3$PerformanceRating[data3$PerformanceRating == 2] <- 'Good'
data3$PerformanceRating[data3$PerformanceRating == 3] <- 'Excellent'
data3$PerformanceRating[data3$PerformanceRating == 4] <- 'Outstanding'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 1] <- 'Low'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 2] <- 'Medium'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 3] <- 'High'
data3$RelationshipSatisfaction[data3$RelationshipSatisfaction == 4] <- 'Very High'
data3$WorkLifeBalance[data3$WorkLifeBalance == 1] <- 'Bad'
data3$WorkLifeBalance[data3$WorkLifeBalance == 2] <- 'Good'
data3$WorkLifeBalance[data3$WorkLifeBalance == 3] <- 'Better'
data3$WorkLifeBalance[data3$WorkLifeBalance == 4] <- 'Best'

# Set variables to factors
data3_original <- data3
data3$Attrition <- factor(data3$Attrition,levels=c("Yes","No")) # Making "Yes" the true value for bayes
data3$BusinessTravel <- as.factor(data3$BusinessTravel)
data3$Department <- as.factor(data3$Department)
data3$EducationField <- as.factor(data3$EducationField)
data3$Gender <- as.factor(data3$Gender)
data3$JobRole <- as.factor(data3$JobRole)
data3$MaritalStatus <- as.factor(data3$MaritalStatus)
data3$OverTime <- as.factor(data3$OverTime)
data3$Education <- factor(data3$Education,levels=c("Below College","College","Bachelor","Master","Doctor"))
data3$EnvironmentSatisfaction <- factor(data3$EnvironmentSatisfaction,levels=c("Low","Medium","High","Very High"))
data3$JobInvolvement <- factor(data3$JobInvolvement,levels=c("Low","Medium","High","Very High"))
data3$JobLevel <- as.factor(data3$JobLevel)
data3$JobSatisfaction <- factor(data3$JobSatisfaction,levels=c("Low","Medium","High","Very High"))
data3$PerformanceRating <- factor(data3$PerformanceRating,levels=c("Low","Good","Excellent","Outstanding"))
data3$RelationshipSatisfaction <- factor(data3$RelationshipSatisfaction,levels=c("Low","Medium","High","Very High"))
data3$StockOptionLevel <- as.factor(data3$StockOptionLevel)
data3$WorkLifeBalance <- factor(data3$WorkLifeBalance,levels=c("Bad","Good","Better","Best"))

# Make a YearsAtCompany_Capped variable and cap it at 30
data3$YearsAtCompany_Capped <- pmin(data3$YearsAtCompany,30)

# Create lists of the variables
# List all the categorical and numeric variables
cat_vars <- c('Attrition','BusinessTravel','Department','Education','EducationField',
              'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole',
              'JobSatisfaction','MaritalStatus','OverTime','PerformanceRating',
              'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance')
num_vars <- c('Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyRate',
              'NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
              'TrainingTimesLastYear', 'YearsAtCompany','YearsInCurrentRole',
              'YearsSinceLastPromotion','YearsWithCurrManager', 'YearsAtCompany_Capped')

# Create square vars
nvars <- length(num_vars)
squared_vars <- c()
data3_no_squared <- data3
for (ii in 1:nvars){
  var <- num_vars[ii]
  var_squared <- paste(var,"_squared",sep="")
  data3[,var_squared]<-data3[,var]*data3[,var]
  squared_vars <- c(squared_vars,var_squared)
}

# Get onehot encoded variables
before_onehot <- data3
data3 <- one_hot(as.data.table(data3))
data3 <- as.data.frame(data3)
nvar <- length(cat_vars)
for (ii in 1:nvar){
  var <- cat_vars[ii]
  data3[,var] <- before_onehot[,var]
}
names(data3) <- gsub(' ','_',names(data3))
names(data3) <- gsub('&','AND',names(data3))
names(data3) <- gsub('-','_',names(data3))
col_names <- names(data3)
onehot_vars <- col_names[grepl('_', names(data3))]
onehot_vars <- onehot_vars[!grepl('squared', onehot_vars)]

# Make cross products of variables
data3_before_cross <- data3
to_cross_vars <- c(num_vars,onehot_vars)
nvars <- length(to_cross_vars)
cross_vars <- c()
for (ii in 1:nvars){
  for(jj in 1:nvars){
    if (ii<jj){
      var1 <- to_cross_vars[ii]
      var2 <- to_cross_vars[jj]
      new_col <- data3[,var1]*data3[,var2]
      if(sum(new_col)!=0){
        new_var <- paste(var1,var2,sep="X")
        data3[,new_var]<-new_col
        cross_vars <- c(cross_vars,new_var)
      }
    }
  }
}

# Combine all the variables
all_vars <- c(cat_vars, num_vars, squared_vars, onehot_vars, cross_vars)

# Smaller model
formula_str <- 'MonthlyIncome ~ JobLevel+TotalWorkingYears+TotalWorkingYears_squared+JobRole_Manager+TotalWorkingYearsXYearsAtCompany_Capped+TotalWorkingYearsXBusinessTravel_Travel_Rarely+Attrition_NoXJobLevel_4+AgeXJobRole_Research_Director+JobRole_Research_Director+YearsAtCompany_CappedXJobRole_Manager'
formula_obj <- as.formula(formula_str)
model <- lm(formula_obj, data = data)
pred <- predict(model, data3)
data3$MonthlyIncome <- pred
write.xlsx(data3, 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Predictions/MonthlyIncome_Model_1.xlsx',colNames = TRUE)

# Larger model
formula_str <- 'MonthlyIncome ~  JobLevel+TotalWorkingYears+AgeXTotalWorkingYears+TotalWorkingYears_squared+JobRole_Manager+TotalWorkingYearsXYearsAtCompany_Capped+TotalWorkingYearsXBusinessTravel_Travel_Rarely+JobLevel_1XPerformanceRating_Excellent+Attrition_NoXJobLevel_4+AgeXJobRole_Research_Director+TotalWorkingYearsXJobRole_Research_Director+JobRole_Research_Director+MonthlyRateXJobRole_Manager+YearsAtCompany_Capped_squared+Attrition_NoXJobRole_Research_Director+YearsAtCompany_CappedXJobRole_Manager+YearsAtCompanyXYearsAtCompany_Capped+YearsAtCompany_CappedXJobLevel_4+MonthlyRateXYearsAtCompany+YearsSinceLastPromotionXYearsAtCompany_Capped+Education_BachelorXJobLevel_1+JobRole_Laboratory_Technician+AgeXJobLevel_3+AgeXJobRole_Research_Scientist+JobLevel_1XJobRole_Laboratory_Technician+TrainingTimesLastYearXJobRole_Research_Scientist+HourlyRateXJobRole_Laboratory_Technician+AgeXJobRole_Laboratory_Technician+MonthlyRateXYearsInCurrentRole+JobLevel_1XRelationshipSatisfaction_High+TrainingTimesLastYearXJobLevel_3+TotalWorkingYearsXEducationField_Medical+DailyRateXJobLevel_3+Attrition_YesXJobLevel_1+DailyRateXYearsInCurrentRole+DistanceFromHomeXYearsAtCompany_Capped+TotalWorkingYearsXJobRole_Research_Scientist+TotalWorkingYearsXJobRole_Laboratory_Technician+YearsInCurrentRoleXDepartment_Research_AND_Development+YearsInCurrentRoleXJobRole_Research_Scientist+YearsSinceLastPromotionXDepartment_Research_AND_Development+YearsAtCompanyXJobRole_Laboratory_Technician+JobRole_Research_ScientistXOverTime_Yes+Education_BachelorXJobLevel_3+YearsInCurrentRoleXGender_Male+YearsInCurrentRoleXJobRole_Laboratory_Technician+DistanceFromHomeXYearsWithCurrManager+DistanceFromHomeXJobRole_Sales_Representative+YearsSinceLastPromotionXJobRole_Research_Scientist+DistanceFromHomeXYearsSinceLastPromotion+YearsWithCurrManagerXJobSatisfaction_Medium+YearsSinceLastPromotionXJobRole_Laboratory_Technician+YearsSinceLastPromotionXRelationshipSatisfaction_Medium+TotalWorkingYearsXJobRole_Sales_Representative+TotalWorkingYearsXJobLevel_2+MonthlyRateXJobLevel_2+YearsSinceLastPromotionXJobRole_Sales_Executive'
formula_obj <- as.formula(formula_str)
model <- lm(formula_obj, data = data)
pred <- predict(model, data3)
data3$MonthlyIncome <- pred
write.xlsx(data3, 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Predictions/MonthlyIncome_Model_2.xlsx',colNames = TRUE)
