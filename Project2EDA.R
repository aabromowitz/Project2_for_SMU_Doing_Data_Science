# The purpose of this script is a test bed for the Project 2 EDA.
# It looks like the overall goal is:
#   1. determine top 3 features for Attrition
#   2. See if there are any trends based on the Job Role
#   3. Come up with a model to predict Salary
# It says that the models should stick to KNN, Naive Bayes, and Linear

# Dataset came from here on Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# It has some info about what the numeric values for things like education, etc. mean

###########Beginning Logic###############################

# Packages
library(tidyverse)
# install.packages("RCurl")
library(RCurl)
library(e1071)
library(caret)
library(mltools)
library(data.table)
library(EMT) # install.packages('EMT')
library(openxlsx) # install.packages('openxlsx')
library(class)
library(xgboost) # install.packages("xgboost")
library(shiny)

# Pull in data
data <- read.table(textConnection(getURL(
  "https://s3.us-east-2.amazonaws.com/ddsproject1/CaseStudy2-data.csv"
)), sep=",", header=TRUE)

# Reset some of the numeric looking variables to words.
data$Education[data$Education == 1] <- 'Below College'
data$Education[data$Education == 2] <- 'College'
data$Education[data$Education == 3] <- 'Bachelor'
data$Education[data$Education == 4] <- 'Master'
data$Education[data$Education == 5] <- 'Doctor'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 1] <- 'Low'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 2] <- 'Medium'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 3] <- 'High'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 4] <- 'Very High'
data$JobInvolvement[data$JobInvolvement == 1] <- 'Low'
data$JobInvolvement[data$JobInvolvement == 2] <- 'Medium'
data$JobInvolvement[data$JobInvolvement == 3] <- 'High'
data$JobInvolvement[data$JobInvolvement == 4] <- 'Very High'
data$JobSatisfaction[data$JobSatisfaction == 1] <- 'Low'
data$JobSatisfaction[data$JobSatisfaction == 2] <- 'Medium'
data$JobSatisfaction[data$JobSatisfaction == 3] <- 'High'
data$JobSatisfaction[data$JobSatisfaction == 4] <- 'Very High'
data$PerformanceRating[data$PerformanceRating == 1] <- 'Low'
data$PerformanceRating[data$PerformanceRating == 2] <- 'Good'
data$PerformanceRating[data$PerformanceRating == 3] <- 'Excellent'
data$PerformanceRating[data$PerformanceRating == 4] <- 'Outstanding'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 1] <- 'Low'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 2] <- 'Medium'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 3] <- 'High'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 4] <- 'Very High'
data$WorkLifeBalance[data$WorkLifeBalance == 1] <- 'Bad'
data$WorkLifeBalance[data$WorkLifeBalance == 2] <- 'Good'
data$WorkLifeBalance[data$WorkLifeBalance == 3] <- 'Better'
data$WorkLifeBalance[data$WorkLifeBalance == 4] <- 'Best'

# Set variables to factors
data_original <- data
data$Attrition <- factor(data$Attrition,levels=c("Yes","No")) # Making "Yes" the true value for bayes
data$BusinessTravel <- as.factor(data$BusinessTravel)
data$Department <- as.factor(data$Department)
data$EducationField <- as.factor(data$EducationField)
data$Gender <- as.factor(data$Gender)
data$JobRole <- as.factor(data$JobRole)
data$MaritalStatus <- as.factor(data$MaritalStatus)
data$OverTime <- as.factor(data$OverTime)
data$Education <- factor(data$Education,levels=c("Below College","College","Bachelor","Master","Doctor"))
data$EnvironmentSatisfaction <- factor(data$EnvironmentSatisfaction,levels=c("Low","Medium","High","Very High"))
data$JobInvolvement <- factor(data$JobInvolvement,levels=c("Low","Medium","High","Very High"))
data$JobLevel <- as.factor(data$JobLevel)
data$JobSatisfaction <- factor(data$JobSatisfaction,levels=c("Low","Medium","High","Very High"))
data$PerformanceRating <- factor(data$PerformanceRating,levels=c("Low","Good","Excellent","Outstanding"))
data$RelationshipSatisfaction <- factor(data$RelationshipSatisfaction,levels=c("Low","Medium","High","Very High"))
data$StockOptionLevel <- as.factor(data$StockOptionLevel)
data$WorkLifeBalance <- factor(data$WorkLifeBalance,levels=c("Bad","Good","Better","Best"))

# Make a YearsAtCompany_Capped variable and cap it at 30
data$YearsAtCompany_Capped <- pmin(data$YearsAtCompany,30)

# Create lists of the variables
# List all the categorical and numeric variables
cat_vars <- c('BusinessTravel','Department','Education','EducationField',
              'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole',
              'JobSatisfaction','MaritalStatus','OverTime','PerformanceRating',
              'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance')
num_vars <- c('Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome',
              'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
              'TrainingTimesLastYear','YearsAtCompany','YearsAtCompany_Capped',
              'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager')

# Create square vars
nvars <- length(num_vars)
squared_vars <- c()
data_no_squared <- data
for (ii in 1:nvars){
  var <- num_vars[ii]
  var_squared <- paste(var,"_squared",sep="")
  data[,var_squared]<-data[,var]*data[,var]
  squared_vars <- c(squared_vars,var_squared)
}

# Get onehot encoded variables
before_onehot <- data
data <- one_hot(as.data.table(data))
data <- as.data.frame(data)
data$Attrition <- before_onehot$Attrition
data$Attrition_Yes <- c()
data$Attrition_No <- c()
nvar <- length(cat_vars)
for (ii in 1:nvar){
  var <- cat_vars[ii]
  data[,var] <- before_onehot[,var]
}
names(data) <- gsub(' ','_',names(data))
names(data) <- gsub('&','AND',names(data))
names(data) <- gsub('-','_',names(data))
col_names <- names(data)
onehot_vars <- col_names[grepl('_', names(data))]
onehot_vars <- onehot_vars[!grepl('squared', onehot_vars)]

# Make cross products of variables
data_before_cross <- data
to_cross_vars <- c(num_vars,onehot_vars)
nvars <- length(to_cross_vars)
cross_vars <- c()
for (ii in 1:nvars){
  for(jj in 1:nvars){
    if (ii<jj){
      var1 <- to_cross_vars[ii]
      var2 <- to_cross_vars[jj]
      new_col <- data[,var1]*data[,var2]
      if(sum(new_col)!=0){
        new_var <- paste(var1,var2,sep="X")
        data[,new_var]<-new_col
        cross_vars <- c(cross_vars,new_var)
      }
    }
  }
}

# Combine all the variables
all_vars <- c(cat_vars, num_vars, squared_vars, onehot_vars, cross_vars)

#########################################################

# Pull in data
data <- read.csv("https://raw.githubusercontent.com/BivinSadler/MSDS_6306_Doing-Data-Science/Master/Unit%2014%20and%2015%20Case%20Study%202/CaseStudy2-data.csv", header=TRUE)
dim(data) # 870  36
summary(data)
data %>% 
  ggplot(aes(Attrition,Age)) +
  geom_point()
cor(data$Attrition, data$Age) # aw, doesn't work for non-numeric

# Pull in data from s3
Sys.setenv("AWS_ACCESS_KEY_ID" = "AKIARXUIWXWHQS6F23XS",
           "AWS_SECRET_ACCESS_KEY" = "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")
data <- read.table(textConnection(getURL(
  "https://s3.us-east-2.amazonaws.com/ddsproject1/CaseStudy2-data.csv"
)), sep=",", header=TRUE)
climate <- read.table(textConnection(getURL(
  "https://cgiardata.s3-us-west-2.amazonaws.com/ccafs/amzn.csv"
)), sep=",", header=TRUE)
creativity <- read.table(textConnection(getURL(
  "https://s3.us-east-2.amazonaws.com/ds6306.unit13/Creativity.csv"
)), sep=",", header=TRUE)

# A lot of the variables look numeric, but are kinda categorical (Ex: Education)
# First try making those factors
class(data1_1$Attrition) # character
data1_2 <- data1_1
data1_2$Attrition <- as.factor(data1_2$Attrition)
class(data1_2$Age)
data1_2$BusinessTravel <- as.factor(data1_2$BusinessTravel)
data1_2$Department <- as.factor(data1_2$Department)
data1_2$EducationField <- as.factor(data1_2$EducationField)
data1_2$Gender <- as.factor(data1_2$Gender)
data1_2$JobRole <- as.factor(data1_2$JobRole)
data1_2$MaritalStatus <- as.factor(data1_2$MaritalStatus)
unique(data1_2$Over18) # Ha, only "Y"
data1_2$OverTime <- as.factor(data1_2$OverTime)
chisq.test(data1_2$Attrition, data1_2$BusinessTravel) # 0.04993, if < 0.5 then independent

# Update certain categorical variables so that they are word values instead of numbers
data$Education[data$Education == 1] <- 'Below College'
data$Education[data$Education == 2] <- 'College'
data$Education[data$Education == 3] <- 'Bachelor'
data$Education[data$Education == 4] <- 'Master'
data$Education[data$Education == 5] <- 'Doctor'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 1] <- 'Low'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 2] <- 'Medium'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 3] <- 'High'
data$EnvironmentSatisfaction[data$EnvironmentSatisfaction == 4] <- 'Very High'
data$JobInvolvement[data$JobInvolvement == 1] <- 'Low'
data$JobInvolvement[data$JobInvolvement == 2] <- 'Medium'
data$JobInvolvement[data$JobInvolvement == 3] <- 'High'
data$JobInvolvement[data$JobInvolvement == 4] <- 'Very High'
data$JobSatisfaction[data$JobSatisfaction == 1] <- 'Low'
data$JobSatisfaction[data$JobSatisfaction == 2] <- 'Medium'
data$JobSatisfaction[data$JobSatisfaction == 3] <- 'High'
data$JobSatisfaction[data$JobSatisfaction == 4] <- 'Very High'
data$PerformanceRating[data$PerformanceRating == 1] <- 'Low'
data$PerformanceRating[data$PerformanceRating == 2] <- 'Good'
data$PerformanceRating[data$PerformanceRating == 3] <- 'Excellent'
data$PerformanceRating[data$PerformanceRating == 4] <- 'Outstanding'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 1] <- 'Low'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 2] <- 'Medium'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 3] <- 'High'
data$RelationshipSatisfaction[data$RelationshipSatisfaction == 4] <- 'Very High'
data$WorkLifeBalance[data$WorkLifeBalance == 1] <- 'Bad'
data$WorkLifeBalance[data$WorkLifeBalance == 2] <- 'Good'
data$WorkLifeBalance[data$WorkLifeBalance == 3] <- 'Better'
data$WorkLifeBalance[data$WorkLifeBalance == 4] <- 'Best'

# There's still columns like JobLevel which still seem very categorical to me
data1_2$Education <- as.factor(data1_2$Education)
unique(data1_2$Education)
data1_2$EnvironmentSatisfaction <- as.factor(data1_2$EnvironmentSatisfaction)
data1_2$JobInvolvement <- as.factor(data1_2$JobInvolvement)
data1_2$JobLevel <- as.factor(data1_2$JobLevel)
data1_2$JobSatisfaction <- as.factor(data1_2$JobSatisfaction)
length(data1_2$PerformanceRating[data1_2$PerformanceRating==3]) # 738
length(data1_2$PerformanceRating[data1_2$PerformanceRating==4]) # 132
data1_2$PerformanceRating <- as.factor(data1_2$PerformanceRating)
data1_2$RelationshipSatisfaction <- as.factor(data1_2$RelationshipSatisfaction)
unique(data1_2$StandardHours) # only 80
data1_2$StockOptionLevel <- as.factor(data1_2$StockOptionLevel)
data1_2$WorkLifeBalance <- as.factor(data1_2$WorkLifeBalance)

# Are there any missing values?
any(is.na(data_original)) # False, that's nice

# List all the categorical and numeric variables
cat_vars <- c('Department','Education','EducationField','EnvironmentSatisfaction',
              'Gender','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus',
              'OverTime','PerformanceRating','RelationshipSatisfaction','StockOptionLevel',
              'WorkLifeBalance')
num_vars <- c('Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate',
              'NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
              'TrainingTimesLastYear', 'YearsAtCompany','YearsInCurrentRole',
              'YearsSinceLastPromotion','YearsWithCurrManager')
all_vars <- c(cat_vars, num_vars)

# Try creating a histogram and saving it off
# All the histograms look the same for both Yes and No
png("C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Plots_Attrition/YearsWithCurrManager_Hist.png")
data %>% ggplot(aes(x=YearsWithCurrManager)) + 
  geom_histogram() + 
  facet_wrap(~ Attrition, ncol=1, scales = 'free')
dev.off()

# Try getting the percentages for each value in the categorical variable for both Yes and No
col = 'WorkLifeBalance'
vals <- unique(as.character(data[,col]))
height <- length(vals)
prop_table = matrix(rep(0, times=2*height), ncol=2, byrow=TRUE)
rownames(prop_table) <- vals
colnames(prop_table) <- c('Yes','No')
for (ii in 1:height){
  prop_table[ii,1] = sum(data$Attrition[data[col] == vals[ii]]=="Yes")
  prop_table[ii,2] = sum(data$Attrition[data[col] == vals[ii]]=="No")
}

# Try creating a histogram for each numeric variable
for (ii in num_vars){
  png(paste("C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Plots_Attrition/",ii,"_Hist.png"))
  data %>% ggplot(aes(x=.data[[ii]])) + # This part didn't feel like working, so I gave up
    geom_histogram() + 
    facet_wrap(~ Attrition, ncol=1, scales = 'free')
  dev.off()
}

# Try creating a histogram for YearsSinceLastPromotion
data %>% ggplot(aes(x=YearsSinceLastPromotion)) +
  geom_histogram(fill = "#13294b", binwidth =1)  + ylab('Count') + 
  xlab('Years Since Last Promotion') + ggtitle('Histogram of Years Since Last Promotion') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Try creating a histogram for YearsAtCompany
data %>% ggplot(aes(x=YearsAtCompany)) +
  geom_histogram(fill = "#13294b", binwidth =1)  + ylab('Count') + 
  xlab('Years at Company') + ggtitle('Histogram of Years at Company') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Try creating a histogram for YearsAtCompany_Capped
data %>% ggplot(aes(x=YearsAtCompany)) +
  geom_histogram(fill = "#13294b", binwidth =1)  + ylab('Count') + 
  xlab('Years at Company') + ggtitle('Histogram of Years at Company - Capped') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Try doing a linear regression
model <- lm(Attrition ~ Age, data = data) # Oh yeah, can't really do this for a categorical value
summary(model)

# Try doing a bayes model
library(e1071)
nb_model <- naiveBayes(Attrition ~ Age, data = data)
summary(nb_model) # This doesn't appear to be super useful
preds <- predict(nb_model, newdata=data)
library(caret)
cmBayes <- confusionMatrix(table(preds,data$Attrition))
cmBayes$overall["Accuracy"] # 0.839 # total correct / total
cmBayes$byClass["Sensitivity"] # 0, true yes / num yes
cmBayes$byClass["Specificity"] # 1, true no / num no, guessing all no
cmBayes$byClass["Precision"] # NA, true yes / num yes predicted, no yes predicted
cmBayes$byClass["Recall"] # 0, true yes / num yes
cmBayes$byClass["F1"] # NA, 2*prec*recall/(prec+recall)

# See the chi squared values for each of the categorical variables
# Highest 3 are OverTime, StockOptionLevel, and JobRole
numVars = length(cat_vars)
pVal = numeric(numVars)
for (ii in 1:numVars){
  col = cat_vars[ii]
  result <- chisq.test(data$Attrition, as.character(data[,col]))
  pVal[ii] = result$p.value
}
cat_vars_df<-data.frame(Variable=cat_vars)
cat_vars_df$PVal <- pVal

# For categorical and numeric variables, you can use something called cor.test
data$Attrition_num <- 0 # No
data$Attrition_num[data$Attrition == "Yes"] <- 1
test = cor.test(data$Attrition_num, data$Age) 
# Since negative, Age is higher for "No"
# Since p-value is < 0.05, they are different

# Try it for all the numerical variables
numVars = length(num_vars)
pVal = numeric(numVars)
for (ii in 1:numVars){
  col = num_vars[ii]
  result <- cor.test(data$Attrition_num, as.numeric(data[,col]))
  pVal[ii] = result$p.value
}
num_vars_df<-data.frame(Variable=num_vars)
num_vars_df$PVal <- pVal
# The lowest three p values are TotalWorkingYears, YearsInCurrentRole, MonthlyIncome
# The p-values aren't as low as the lowest categorical data p-values though

# Loop through all the variables to see what all the prediction accuracies are
all_vars_df = rbind.data.frame(cat_vars_df, num_vars_df)
all_vars_df$Accuracy <- 0
all_vars_df$Sensitivity <- 0
all_vars_df$Specificity <- 0
all_vars_df$Precision <- 0
all_vars_df$Recall <- 0
all_vars_df$F1 <- 0
numVars <- nrow(all_vars_df)
for (ii in 1:numVars){
  var = all_vars_df$Variable[ii]
  formula_str <- paste("Attrition ~", var)
  formula_obj <- as.formula(formula_str)
  nb_model <- naiveBayes(formula_obj, data = data)
  preds <- predict(nb_model, newdata=data)
  cmBayes <- confusionMatrix(table(preds,data$Attrition))
  all_vars_df$Accuracy[ii] <- cmBayes$overall["Accuracy"] # total correct / total
  all_vars_df$Sensitivity[ii] <- cmBayes$byClass["Sensitivity"] # true yes / num yes
  all_vars_df$Specificity[ii] <- cmBayes$byClass["Specificity"] # true no / num no
  all_vars_df$Precision[ii] <- cmBayes$byClass["Precision"] # true yes / num yes predicted
  all_vars_df$Recall[ii] <- cmBayes$byClass["Recall"] # true yes / num yes
  all_vars_df$F1[ii] <- cmBayes$byClass["F1"] # 2*prec*recall/(prec+recall)
}
# Interesting, it just chose "No" for all the predictions, for each column
# I guess no individual value had more "Yes" than "No", so that makes some sense

# What happens if you do a naive bayes with all the variables?
nb_model <- naiveBayes(Attrition ~ ., data = data)
preds <- predict(nb_model, newdata=data)
cmBayes <- confusionMatrix(table(preds,data$Attrition))
# Oh good, 100%

# Is there a way to combine naive bayes models?
indices = sample(seq(1:nrow(data)),round(.5*nrow(data)))
first_half = data[indices,]
second_half = data[-indices,]
nb_model1 <- naiveBayes(Attrition ~ ., data = first_half)
nb_model2 <- naiveBayes(Attrition ~ ., data = second_half)
nb_model <- nb_model1 + nb_model2 # This doesn't work
# Chat GPT says you can combine the predictions, so predict X number of times, and then choose the most common

# what is the validation if you try the approach 10 times
metric <- 0
num_tests <- 10
seed <- 1

set.seed(seed)
for(ii in 1:num_tests){
  num_vars <- length(all_vars_df$Variable)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  formula_str <- paste("Attrition ~ ", paste(all_vars_df$Variable,collapse="+"))
  formula_obj <- as.formula(formula_str)
  nb_models <- vector("list", num_ensemble)
  for (ii in 1:num_ensemble) {
    indices = sample(seq(1:nrow(train)),round(prop_test*nrow(train)))
    test <- train[indices,]
    train2 <- train[-indices,]
    nb_models[[ii]] <- naiveBayes(formula_obj, data = train2)
  }
  predictions <- sapply(nb_models, function(model) predict(model, validation))
  final_predictions <- apply(predictions, 1, function(x) {
    # Use majority vote to determine the final prediction
    tab <- table(x)
    factor(names(tab)[which.max(tab)],levels=c("Yes","No")) # Assuming factors where "Yes" is 1 and "No" is 0
  })
  cmBayes <- confusionMatrix(table(final_predictions,validation$Attrition))
  metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
}
metric = metric / num_tests # oh, it gets worse when you don't include "Attrition"

# Before doing it programatically, see what happens if you remove a 

# Try just using a few of the good vars
seed <- 2
prop_test <- .1
num_ensemble <- 5
vars_to_use <- c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel",
                 "MaritalStatus")
nb_vars <- prune_vars_nb(data, vars_to_use, seed, prop_test, num_ensemble)

# Loop through all the variables, remove the least predictive, then see how the model does
seed <- 5
prop_test <- .1
num_ensemble <- 5
nb_vars <- prune_vars_nb(data, all_vars_df$Variable, seed, prop_test, num_ensemble)

# Try getting correlation values between each variables
# There are very few variables with high correlations
nvars <- ncol(data)
corrs <- matrix(0, nvars, nvars)
rownames(corrs) <- names(data)
colnames(corrs) <- names(data)
for (ii in 1:nvars){
  for (jj in 1:nvars){
    col1 <- unclass(data[,ii])
    col2 <- unclass(data[,jj])
    if (length(unique(col1)) == 1 || length(unique(col2)) == 1){
      corrs[ii,jj] <- 0
    } else{
      corrs[ii,jj] <- cor(col1,col2)
    }
  }
}

data$ID <- c()
data$EmployeeCount <- c()
data$EmployeeNumber <- c()
data$StandardHours <- c()
data$Over18 <- c()

# Making the col matrix and saving it off
nvars <- ncol(data)
corrs <- data.frame(matrix(ncol = 3, nrow = nvars*nvars))
colnames(corrs) <- c('var1', 'var2', 'corr')
rowNum <- 1
for (ii in 1:nvars){
  for (jj in 1:nvars){
    var1 <- unclass(data[,ii])
    var2 <- unclass(data[,jj])
    if (length(unique(var1)) == 1 || length(unique(var2)) == 1){
      corr <- 0
    } else{
      corr <- cor.test(var1,var2)
    }
    corrs$var1[rowNum] <- names(data[ii])
    corrs$var2[rowNum] <- names(data[jj])
    corrs$corr[rowNum] <- as.numeric(corr$estimate)
    rowNum <- rowNum + 1
  }
}
write.xlsx(corrs,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/corrs.xlsx',colNames = TRUE)

# Plotting Martial Status and Stock Option Level
sum(data$StockOptionLevel[data$MaritalStatus=='Single']==0) # 269, 100.0% vs 43.6%
sum(data$StockOptionLevel[data$MaritalStatus=='Single']==1) # 0, 0% vs 40.8%
sum(data$StockOptionLevel[data$MaritalStatus=='Single']==2) # 0, 0% vs 9.3%
sum(data$StockOptionLevel[data$MaritalStatus=='Single']==3) # 0, 0% vs 6.3%
sum(data$StockOptionLevel[data$MaritalStatus=='Divorced']==0) # 5, 2.6% vs 43.6%
sum(data$StockOptionLevel[data$MaritalStatus=='Divorced']==1) # 119, 62.3% vs 40.8%
sum(data$StockOptionLevel[data$MaritalStatus=='Divorced']==2) # 34, 17.8% vs 9.3%
sum(data$StockOptionLevel[data$MaritalStatus=='Divorced']==3) # 33, 17.3% vs 6.3%
sum(data$StockOptionLevel[data$MaritalStatus=='Married']==0) # 105, 25.6% vs 43.6%
sum(data$StockOptionLevel[data$MaritalStatus=='Married']==1) # 236, 57.6% vs 40.8% 
sum(data$StockOptionLevel[data$MaritalStatus=='Married']==2) # 47, 11.5% vs 9.3%
sum(data$StockOptionLevel[data$MaritalStatus=='Married']==3) # 22, 5.4% vs 6.3%
data %>% ggplot(aes(x=StockOptionLevel, fill=MaritalStatus)) +
  geom_bar() + xlab('Stock Option Level') + ylab('Count') + 
  ggtitle('Stock Option Level by Marital Status') +
  scale_fill_manual("Marital Status", 
    values = c("Single" = "#13294b", "Divorced" = "lightblue", "Married" = "blue")) +
    theme(plot.background = element_rect("#dbdfdf"),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
          axis.text.y = element_text(color = "#13294b",size = 14), 
          axis.title.x = element_text(color = "#13294b",size = 16), 
          axis.title.y =element_text(color = "#13294b",size = 16), 
          plot.title  = element_text(color = "#13294b",size = 18))

# How does this compare to Attrition?
data %>% ggplot(aes(x=StockOptionLevel, fill=Attrition)) +
  geom_bar() + xlab('Stock Option Level') + ylab('Count') + 
  ggtitle('Stock Option Level by Attrition') +
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Attrition and OverTime
data %>% ggplot(aes(x=OverTime, fill=Attrition)) +
  geom_bar() + xlab('Overtime') + ylab('Count') + 
  ggtitle('Overtime by Attrition') +
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Try plotting percentage
data %>% ggplot(aes(x=OverTime, fill=Attrition)) +
  geom_bar(position = "fill") + xlab('Overtime') + ylab('Percent') + 
  ggtitle('Overtime by Attrition Percentage') +
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  scale_y_continuous(labels = scales::percent_format(scale = 100)) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Age and Education
data %>% ggplot(aes(x=Age)) +
  geom_histogram(fill="#13294b") + xlab('Age') + ylab('Count') + 
  ggtitle('Age vs Education Level') +
  facet_wrap(~ Education, ncol=1) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 7), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot average
data %>% group_by(Education) %>% summarise(mean=mean(Age)) %>%
  ggplot(aes(x=Education,y=mean)) +
  geom_point(size=3,color="#13294b") + xlab('Education Level') + ylab('Average Age') + ylim(30,40)+
  ggtitle('Average Age vs Education Level')+
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))
  
# Plot Attrition vs JobInvolvement
data %>% ggplot(aes(x=JobInvolvement,fill=Attrition)) +
  geom_bar(position="fill") + xlab('Job Involvement') + ylab('Percent') + 
  ggtitle('Attrition based on Job Involvement') +
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  scale_y_continuous(labels = scales::percent_format(scale = 100)) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Attrition vs JobLevel
data %>% ggplot(aes(x=JobLevel,fill=Attrition)) +
  geom_bar(position="fill") + xlab('Job Level') + ylab('Percent') + 
  ggtitle('Attrition based on Job Level') +
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  scale_y_continuous(labels = scales::percent_format(scale = 100)) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Attrition vs TotalWorkingYears
data %>% ggplot(aes(x=TotalWorkingYears)) +
  geom_histogram(fill="#13294b") + xlab('Attrition') + ylab('Count') + 
  ggtitle('Attrition vs Total Working Years') +
  facet_wrap(~ Attrition, ncol=1, scales='free') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 7), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Attrition vs MonthlyIncome
data %>% ggplot(aes(x=MonthlyIncome)) +
  geom_histogram(fill="#13294b") + xlab('Attrition') + ylab('Count') + 
  ggtitle('Attrition vs Monthly Income') +
  facet_wrap(~ Attrition, ncol=1, scales='free') +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, color = "#13294b",size = 10),
        axis.text.y = element_text(color = "#13294b",size = 7), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 18))

# Plot Attrition vs JobLevel_1XOverTime_Yes
data %>% ggplot(aes(x=JobLevel_1XOverTime_Yes,fill=Attrition)) +
  geom_bar(position="fill") + xlab('Job Level = 1 X Overtime = Yes') + ylab('Percent') + 
  ggtitle('Attrition by Derived Variable JobLevel_1XOverTime_Yes') + 
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  scale_y_continuous(labels = scales::percent_format(scale = 100)) +
  scale_x_discrete(breaks=c(0,1),limits=c(0,1)) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=0.5, color = "#13294b",size = 14),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 14))

# Plot Attrition vs OverTime_YesXStockOptionLevel_0
data %>% ggplot(aes(x=OverTime_YesXStockOptionLevel_0,fill=Attrition)) +
  geom_bar(position="fill") + xlab('Overtime = Yes X Stock Option Level = 0') + ylab('Percent') + 
  ggtitle('Attrition by Derived Variable OverTime_YesXStockLevel_0') + 
  scale_fill_manual("Attrition", 
                    values = c("Yes" = "blue", "No" = "#13294b")) +
  scale_y_continuous(labels = scales::percent_format(scale = 100)) +
  scale_x_discrete(breaks=c(0,1),limits=c(0,1)) +
  theme(plot.background = element_rect("#dbdfdf"),
        axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=0.5, color = "#13294b",size = 14),
        axis.text.y = element_text(color = "#13294b",size = 14), 
        axis.title.x = element_text(color = "#13294b",size = 16), 
        axis.title.y =element_text(color = "#13294b",size = 16), 
        plot.title  = element_text(color = "#13294b",size = 14))

# Look at Age vs education
# Avg age goes up with education level
data %>% ggplot(aes(x=Age)) + 
  geom_histogram() + 
  facet_wrap(~ Education, ncol=1, scales = 'free')
mean(data$Age[data$Education == 'Below College']) # 32.37
mean(data$Age[data$Education == 'College']) # 36.1
mean(data$Age[data$Education == 'Bachelor']) # 36.6
mean(data$Age[data$Education == 'Master']) # 39.18
mean(data$Age[data$Education == 'Doctor']) # 39.77

# Look at business travel and environment satisfaction
xtabs( ~ data[,'BusinessTravel'] + data[,'EnvironmentSatisfaction'])
# It looks like travel is associated with high environment satisfaction

# Look at the satisfaction (Environment and Job) between different categories
xtabs( ~ data[,'Gender'] + data[,'EnvironmentSatisfaction'])
cor.test(unclass(data[,'Gender']),unclass(data[,'EnvironmentSatisfaction']))
xtabs( ~ data[,'JobRole'] + data[,'EnvironmentSatisfaction'])
xtabs( ~ data[,'JobRole'] + data[,'JobSatisfaction'])

# Maybe look at p-values between the different variables
# Oddly, nothing was associated highly with Job Satisfaction or Environment Satisfaction
nvars <- ncol(data)
pvals <- matrix(0, nvars, nvars)
rownames(pvals) <- names(data)
colnames(pvals) <- names(data)
for (ii in 1:nvars){
  for (jj in 1:nvars){
    col1 <- unclass(data[,ii])
    col2 <- unclass(data[,jj])
    if (length(unique(col1)) == 1 || length(unique(col2)) == 1){
      pvals[ii,jj] <- 0
    } else{
      test <- cor.test(col1,col2)
      pvals[ii,jj] <- test$p.value
    }
  }
}

# At what level does adding variables make it worse
seed <- 1
num_tests <- 200
prop_test <- .1
thresh <- .6
formula_str <- "Attrition ~ OverTime+MaritalStatus+TotalWorkingYears+JobLevel+YearsInCurrentRole+MonthlyIncome+Age+StockOptionLevel+YearsWithCurrManager+YearsAtCompany+JobRole+DistanceFromHome+JobSatisfaction+WorkLifeBalance+NumCompaniesWorked+MonthlyRate+DailyRate+PerformanceRating"

metric <- 0
num_passed <- 0
set.seed(seed)
for (ii in 1:num_tests){
  formula_obj <- as.formula(formula_str)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  model <- naiveBayes(formula_obj, data = train)
  pred <- predict(model, validation)
  cmBayes <- confusionMatrix(table(pred,validation$Attrition))
  if(cmBayes$byClass["Sensitivity"]>thresh && cmBayes$byClass["Specificity"]>thresh) {
    metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    num_passed <- num_passed + 1
  }
}
metric / num_tests
num_passed

# Does bagging actually help?
seed <- 1
num_tests <- 100
prop_test <- .1
thresh <- .6
formula_str <- "Attrition ~ OverTime+MaritalStatus+TotalWorkingYears+JobLevel+YearsInCurrentRole+MonthlyIncome+Age+StockOptionLevel+YearsWithCurrManager+YearsAtCompany+JobRole+DistanceFromHome+JobSatisfaction+WorkLifeBalance+NumCompaniesWorked+MonthlyRate+DailyRate+PerformanceRating"

set.seed(seed)
metric <- 0
num_passed <- 0
num_ensemble <- 25
for (ii in 1:num_tests){
  formula_obj <- as.formula(formula_str)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  nb_models <- vector("list", num_ensemble)
  for (kk in 1:num_ensemble) {
    indices2 = sample(seq(1:nrow(train)),round(prop_test*nrow(train)))
    test <- train[indices,]
    train2 <- train[-indices,]
    nb_models[[kk]] <- naiveBayes(formula_obj, data = train2)
  }
  predictions <- sapply(nb_models, function(model) predict(model, validation))
  final_predictions <- apply(predictions, 1, function(x) {
    tab <- table(x)
    factor(names(tab)[which.max(tab)],levels=c("Yes","No")) # Assuming factors where "Yes" is 1 and "No" is 0
  })
  cmBayes <- confusionMatrix(table(final_predictions,validation$Attrition))
  if(cmBayes$byClass["Sensitivity"]>thresh && cmBayes$byClass["Specificity"]>thresh) {
    metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    num_passed <- num_passed + 1
  }
}
metric / num_tests
num_passed # Only 40 passed, so it kind of seems to make it worse

# Creating square variables
nvars <- length(num_vars)
for (ii in 1:nvars){
  var <- num_vars[ii]
  var_squared <- paste(var,"_squared",sep="")
  data[,var_squared]<-data[,var]*data[,var]
}

# Does just using all the variables work?
seed <- 1
num_tests <- 200
prop_test <- .1
thresh <- .6
formula_str <- "Attrition ~ ."
# formula_str <- paste("Attrition ~ ", paste(all_vars,collapse="+"))

metric <- 0
num_passed <- 0
set.seed(seed)
for (ii in 1:num_tests){
  formula_obj <- as.formula(formula_str)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  model <- naiveBayes(formula_obj, data = train)
  pred <- predict(model, validation)
  cmBayes <- confusionMatrix(table(pred,validation$Attrition))
  if(cmBayes$byClass["Sensitivity"]>thresh && cmBayes$byClass["Specificity"]>thresh) {
    metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    num_passed <- num_passed + 1
  }
}
metric / num_tests
num_passed
# All original variables was 113, including the squared variables it was 151

# Try doing the go through each at a time logic for squared variables
seed <- 1
prop_test <- .1
num_tests <- 200
vars_df <- add_vars_nb(data, all_vars, seed, prop_test, num_tests)

# Try one-hot encoding
install.packages("mltools")
install.packages("data.table")
library(mltools)
library(data.table)
before_onehot <- data
data <- one_hot(as.data.table(data))
data$Attrition <- before_onehot$Attrition
data$Attrition_Yes <- c()
data$Attrition_No <- c()
col_names <- names(data)
onehot_vars <- col_names[grepl('_', names(data))]
onehot_vars <- onehot_vars[!grepl('squared', onehot_vars)]

# It might be good to put back in the original categorical variables
nvar <- length(cat_vars)
for (ii in 1:nvar){
  var <- cat_vars[ii]
  data[,var] <- before_onehot[,var]
}

# Do onehot variables get good p values?
data <- as.data.frame(data)
var_df <- data.frame(Variables=all_vars)
var_df$p <- 0
var_df$metric <- 0
var_df$num_passed <- 0
var_df$to_use <- 0
var_df$curr_formula <- ""
nvar <- length(all_vars)
for(ii in 1:nvar){
  var<-all_vars[ii]
  test <- cor.test(unclass(data[,'Attrition']),unclass(data[,var]))
  var_df$p[ii] <- test$p.value
}

# After one-hot encoding, try going through all those variables
seed <- 1
prop_test <- .1
num_tests <- 200
vars_df <- add_vars_nb(data, all_vars, seed, prop_test, num_tests)

# I wonder how this result compares to the base result, for a different seed
seed <- 2
num_tests <- 200
prop_test <- .1
thresh <- .6
# formula_str <- paste("Attrition ~ ", paste(c(cat_vars,num_vars),collapse="+"))
formula_str <- "Attrition ~ OverTime+StockOptionLevel_0+JobLevel_1+JobRole_Sales_Representative+JobInvolvement_Low+MaritalStatus+StockOptionLevel_1+MaritalStatus_Single+TotalWorkingYears+JobLevel+YearsInCurrentRole+MonthlyIncome+Age+StockOptionLevel+YearsWithCurrManager+MaritalStatus_Divorced+WorkLifeBalance_Bad+JobRole_Manufacturing_Director+MonthlyIncome_squared+YearsWithCurrManager_squared+EnvironmentSatisfaction_Low+YearsInCurrentRole_squared+TotalWorkingYears_squared+StockOptionLevel_2+JobSatisfaction_Very_High+JobInvolvement_High+DistanceFromHome+Education+HourlyRate"
metric <- 0
num_passed <- 0
set.seed(seed)
for (ii in 1:num_tests){
  formula_obj <- as.formula(formula_str)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  model <- naiveBayes(formula_obj, data = train)
  pred <- predict(model, validation)
  cmBayes <- confusionMatrix(table(pred,validation$Attrition))
  if(cmBayes$byClass["Sensitivity"]>thresh && cmBayes$byClass["Specificity"]>thresh) {
    metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    num_passed <- num_passed + 1
  }
}
metric / num_tests
num_passed # 114 for all normal variables, 174 for updated formula

# Can we make cross products of variables
data_before_cross <- data
to_cross_vars <- c(num_vars,onehot_vars)
nvars <- length(to_cross_vars)
for (ii in 1:nvars){
  for(jj in 1:nvars){
    if (ii<jj){
      var1 <- to_cross_vars[ii]
      var2 <- to_cross_vars[jj]
      new_col <- data[,var1]*data[,var2]
      if(sum(new_col)!=0){
        new_var <- paste(var1,var2,sep="X")
        data[,new_var]<-new_col
      }
    }
  }
}

# What do the p-values of the new variables look like
var_df <- data.frame(Variables=all_vars)
var_df$p <- 0
var_df$metric <- 0
var_df$num_passed <- 0
var_df$to_use <- 0
var_df$curr_formula <- ""
nvar <- length(all_vars)
for(ii in 1:nvar){
  var<-all_vars[ii]
  # print(var)
  test <- cor.test(unclass(data[,'Attrition']),unclass(data[,var]))
  var_df$p[ii] <- test$p.value
}

# Try it with the cross variables, but stop if no change after 50 iterations
seed <- 1
prop_test <- .1
num_tests <- 200
num_no_change <- 50
vars_df <- add_vars_nb(data, all_vars, seed, prop_test, num_tests, num_no_change)

# How does the last model compare with the cross product one
seed <- 3
num_tests <- 200
prop_test <- .1
thresh <- .6
# formula_str <- paste("Attrition ~ ", paste(c(cat_vars,num_vars),collapse="+"))
# formula_str <- "Attrition ~ OverTime+StockOptionLevel_0+JobLevel_1+JobRole_Sales_Representative+JobInvolvement_Low+MaritalStatus+StockOptionLevel_1+MaritalStatus_Single+TotalWorkingYears+JobLevel+YearsInCurrentRole+MonthlyIncome+Age+StockOptionLevel+YearsWithCurrManager+MaritalStatus_Divorced+WorkLifeBalance_Bad+JobRole_Manufacturing_Director+MonthlyIncome_squared+YearsWithCurrManager_squared+EnvironmentSatisfaction_Low+YearsInCurrentRole_squared+TotalWorkingYears_squared+StockOptionLevel_2+JobSatisfaction_Very_High+JobInvolvement_High+DistanceFromHome+Education+HourlyRate"
formula_str <- "Attrition~JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+JobLevel_1XMaritalStatus_Single+PercentSalaryHikeXStockOptionLevel_0+StockOptionLevel_0+DistanceFromHomeXStockOptionLevel_0+TotalWorkingYearsXOverTime_No+MonthlyIncomeXOverTime_No+JobLevel_1+HourlyRateXJobInvolvement_Low+DailyRateXStockOptionLevel_0+PercentSalaryHikeXJobLevel_1+NumCompaniesWorkedXStockOptionLevel_0+AgeXStockOptionLevel_1+OverTime_NoXStockOptionLevel_1+JobLevel_1XPerformanceRating_Excellent+AgeXJobRole_Sales_Representative"
metric <- 0
num_passed <- 0
set.seed(seed)
for (ii in 1:num_tests){
  formula_obj <- as.formula(formula_str)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  model <- naiveBayes(formula_obj, data = train)
  pred <- predict(model, validation)
  cmBayes <- confusionMatrix(table(pred,validation$Attrition))
  if(cmBayes$byClass["Sensitivity"]>thresh && cmBayes$byClass["Specificity"]>thresh) {
    metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    num_passed <- num_passed + 1
  }
}
metric / num_tests
num_passed # onehot was 175, cross products was 188

# How does the cross product model do on the whole dataset?
model <- naiveBayes(formula_obj, data = data)
pred <- predict(model, data)
cmBayes <- confusionMatrix(table(pred,data$Attrition))
# Accuracy is 76.55%, Sensitivity is 79.29%, and Specificity is 76.03%, that's not amazing

# Maybe try the same thing, but not worrying if it gets over the 60% threshold, but if it's maximizing the sum.
seed <- 1
prop_test <- .1
thresh <- .6
num_tests <- 200
num_no_change <- 50
vars_df <- add_vars_nb(data, all_vars, seed, prop_test, thresh, num_tests, num_no_change)

# Try saving off some text
writeLines('hello2','C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Save.txt')

# Now test on a different seed
seed <- 4
num_tests <- 200
prop_test <- .1
thresh <- .6
# formula_str <- "Attrition~JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+JobLevel_1XMaritalStatus_Single+PercentSalaryHikeXStockOptionLevel_0+StockOptionLevel_0+DistanceFromHomeXStockOptionLevel_0+TotalWorkingYearsXOverTime_No+MonthlyIncomeXOverTime_No+JobLevel_1+HourlyRateXJobInvolvement_Low+DailyRateXStockOptionLevel_0+PercentSalaryHikeXJobLevel_1+NumCompaniesWorkedXStockOptionLevel_0+AgeXStockOptionLevel_1+OverTime_NoXStockOptionLevel_1+JobLevel_1XPerformanceRating_Excellent+AgeXJobRole_Sales_Representative"
formula_str <- 'Attrition~JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+JobRole_Sales_RepresentativeXStockOptionLevel_0+MaritalStatus+MonthlyRateXOverTime_Yes+PercentSalaryHikeXJobInvolvement_Low+YearsInCurrentRoleXOverTime_No+DailyRateXStockOptionLevel_1+OverTime_NoXStockOptionLevel_1+JobInvolvement_LowXPerformanceRating_Excellent+OverTime_NoXWorkLifeBalance_Better+JobLevel_1XPerformanceRating_Excellent+TotalWorkingYearsXTrainingTimesLastYear+PerformanceRating_ExcellentXStockOptionLevel_0+JobRole_Laboratory_TechnicianXWorkLifeBalance_Bad+TotalWorkingYearsXStockOptionLevel_1+YearsAtCompanyXOverTime_No+MonthlyIncomeXJobLevel_1+MonthlyIncomeXTrainingTimesLastYear+JobSatisfaction_LowXStockOptionLevel_0+EducationField_Technical_DegreeXStockOptionLevel_0+TrainingTimesLastYearXYearsInCurrentRole+MonthlyIncomeXStockOptionLevel_1+TrainingTimesLastYearXYearsWithCurrManager+StockOptionLevel_0XWorkLifeBalance_Bad+YearsInCurrentRoleXDepartment_Research_AND_Development+MaritalStatus_SingleXPerformanceRating_Outstanding+Department_Research_AND_DevelopmentXJobLevel_2+DailyRateXMaritalStatus_Divorced+YearsInCurrentRoleXJobInvolvement_Low+RelationshipSatisfaction_LowXStockOptionLevel_0+JobInvolvement_HighXMaritalStatus_Divorced+TrainingTimesLastYearXYearsAtCompany+JobRole_Laboratory_TechnicianXStockOptionLevel_0+YearsAtCompany+YearsAtCompanyXDepartment_Research_AND_Development+DailyRateXYearsWithCurrManager+WorkLifeBalance_Bad+Gender_FemaleXOverTime_No+OverTime_NoXRelationshipSatisfaction_Medium+YearsAtCompanyXPerformanceRating_Excellent+MonthlyRateXDepartment_Research_AND_Development+JobSatisfaction_HighXStockOptionLevel_0+OverTime_YesXStockOptionLevel_3+TrainingTimesLastYearXDepartment_Research_AND_Development+EnvironmentSatisfaction_MediumXOverTime_No+MonthlyRateXJobLevel_2+Education_CollegeXRelationshipSatisfaction_Low+JobSatisfaction_Very_HighXStockOptionLevel_1+DailyRateXWorkLifeBalance_Bad'
metric <- 0
num_passed <- 0
set.seed(seed)
for (ii in 1:num_tests){
  formula_obj <- as.formula(formula_str)
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  model <- naiveBayes(formula_obj, data = train)
  pred <- predict(model, validation)
  cmBayes <- confusionMatrix(table(pred,validation$Attrition))
  if(cmBayes$byClass["Sensitivity"]>thresh && cmBayes$byClass["Specificity"]>thresh) {
    metric <- metric + cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    num_passed <- num_passed + 1
  }
}
metric / num_tests
num_passed # num_passed was 187 and 1.4549, metric was 196 and 1.6329

# How does the cross product model do on the whole dataset?
model <- naiveBayes(formula_obj, data = data)
pred <- predict(model, data)
cmBayes <- confusionMatrix(table(pred,data$Attrition))
# Accuracy is 84.71%, Sensitivity is 82.86%, and Specificity is 85.07%, a little better

# I guess try KNN as well, maybe OverTime, MaritalStatus, and TotalWorkingYears

# How many values are in the variables chosen for the 
sum(data[,'JobLevel_1XOverTime_Yes']!=0) # 97
sum(data[,'OverTime_YesXStockOptionLevel_0']!=0) # 116
sum(data[,'JobLevel_1XMaritalStatus_Single']!=0) # 117
sum(data[,'DailyRateXWorkLifeBalance_Bad']!=0) # 48, this is still pretty high

# Look at the 'best' crossed variables
sum(data$Attrition[data$JobLevel_1XOverTime_Yes==1]=='Yes') # 52, 53.6% Yes
sum(data$Attrition[data$JobLevel_1XOverTime_Yes==1]=='No') # 45
sum(data$Attrition[data$JobLevel_1XOverTime_Yes==0]=='Yes') # 88, 11.4% Yes
sum(data$Attrition[data$JobLevel_1XOverTime_Yes==0]=='No') # 685

# What's OverTime
sum(data$Attrition[data$OverTime=='Yes']=='Yes') # 80, 31.7% Yes vs 16.1%
sum(data$Attrition[data$OverTime=='Yes']=='No') # 172
sum(data$Attrition[data$OverTime=='No']=='Yes') # 60, 9.7% Yes vs 16.1%
sum(data$Attrition[data$OverTime=='No']=='No') # 558
prob <- c((80+172)/870*140/870,(80+172)/870*730/870,(60+558)/870*140/870,(60+558)/870*730/870)
actual <- c(80,172,60,558)
multinomial.test(actual, prob) # Didn't seem to work

# Marital Status
sum(data$Attrition[data$MaritalStatus=='Single']=='Yes') # 70, 26.0% Yes vs 16.1%
sum(data$Attrition[data$MaritalStatus=='Single']=='No') # 199
sum(data$Attrition[data$MaritalStatus=='Divorced']=='Yes') # 12, 6.3% Yes vs 16.1%
sum(data$Attrition[data$MaritalStatus=='Divorced']=='No') # 179
sum(data$Attrition[data$MaritalStatus=='Married']=='Yes') # 58, 14.1% Yes vs 16.1%
sum(data$Attrition[data$MaritalStatus=='Married']=='No') # 352

# Total Working Years
mean(data$TotalWorkingYears[data$Attrition=='Yes']) # 8.19
mean(data$TotalWorkingYears[data$Attrition=='No']) # 11.6
mean(data$TotalWorkingYears) # 11.05

# Dr Sadler suggested looking at variables that make sense
sum(data$Attrition[data$JobSatisfaction=='Low']=='Yes') # 38, 21.2% Yes vs 16.1%
sum(data$Attrition[data$JobSatisfaction=='Low']=='No') # 141
sum(data$Attrition[data$JobSatisfaction=='Medium']=='Yes') # 31, 18.7% Yes vs 16.1%
sum(data$Attrition[data$JobSatisfaction=='Medium']=='No') # 135
sum(data$Attrition[data$JobSatisfaction=='High']=='Yes') # 43, 16.9% Yes vs 16.1%
sum(data$Attrition[data$JobSatisfaction=='High']=='No') # 211
sum(data$Attrition[data$JobSatisfaction=='Very High']=='Yes') # 28, 10.3% Yes vs 16.1%
sum(data$Attrition[data$JobSatisfaction=='Very High']=='No') # 243

# How does KNN work for onehot variables?
seed <- 1
prop_test <- .1
set.seed(seed)
indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
validation = data[indices,]
train = data[-indices,]
classifications = knn(train[,c("JobLevel_1XOverTime_Yes","OverTime_YesXStockOptionLevel_0","MaritalStatus_SingleXOverTime_Yes","JobLevel_1XStockOptionLevel_0")], # you might need at least 2 variables
                      validation[,c("JobLevel_1XOverTime_Yes","OverTime_YesXStockOptionLevel_0","MaritalStatus_SingleXOverTime_Yes","JobLevel_1XStockOptionLevel_0")],
                      train$Attrition, prob = TRUE, k = 3)
CM = confusionMatrix(table(validation$Attrition,classifications))
as.numeric(CM$byClass['Sensitivity']) + as.numeric(CM$byClass['Specificity']) # 1.87 on first test, which is really good

# Trying out the knn loop function
seed <- 1
prop_test <- .1
thresh <- .6
num_tests <- 200
num_no_change <- 50
start_formula <- 'JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0'
start_num <- 23
start_best <- 1.56
txt_file <- 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Save_knn.txt'
vars_df <- add_vars_knn(data, all_vars, seed, prop_test, thresh, num_tests, num_no_change, start_formula, start_num, txt_file, start_best)
write.xlsx(vars_df,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/vars_df.xlsx',colNames = TRUE)

# That wasn't very good, what would the intial one I tried look like over 200 trials?
seed <- 1
prop_test <- .1
set.seed(seed)
num_passed <- 0
metric <- 0
for(ii in 1:200){
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  classifications = knn(train[,c("TotalWorkingYears","YearsInCurrentRole")], # you might need at least 2 variables
                        validation[,c("TotalWorkingYears","YearsInCurrentRole")],
                        train$Attrition, prob = TRUE, k = 5)
  CM = confusionMatrix(table(validation$Attrition,classifications))
  sens <- as.numeric(CM$byClass["Sensitivity"])
  if(is.na(sens)) {
    sens<-0
  }
  spec <- as.numeric(CM$byClass["Specificity"])
  if(is.na(spec)){
    spec<-0
  }
  if(sens>thresh && spec>thresh) {
    num_passed <- num_passed + 1
  }
  metric <- metric + sens + spec
}
num_passed
metric/200 # not even 100 are passing and the metric isn't that good, bayes is probably the best option


# Determining how well 3 variables for Naive Bayes works
seed <- 10
prop_test <- .1
num_tests <- 200
# formula_str <- 'Attrition ~ JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+MaritalStatus_SingleXOverTime_Yes'
# formula_str <- 'Attrition ~  JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+JobRole_Sales_RepresentativeXStockOptionLevel_0+MaritalStatus+MonthlyRateXOverTime_Yes+PercentSalaryHikeXJobInvolvement_Low+YearsInCurrentRoleXOverTime_No+DailyRateXStockOptionLevel_1+OverTime_NoXStockOptionLevel_1+JobInvolvement_LowXPerformanceRating_Excellent+OverTime_NoXWorkLifeBalance_Better+JobLevel_1XPerformanceRating_Excellent+TotalWorkingYearsXTrainingTimesLastYear+PerformanceRating_ExcellentXStockOptionLevel_0+JobRole_Laboratory_TechnicianXWorkLifeBalance_Bad+TotalWorkingYearsXStockOptionLevel_1+YearsAtCompanyXOverTime_No+MonthlyIncomeXJobLevel_1+MonthlyIncomeXTrainingTimesLastYear+JobSatisfaction_LowXStockOptionLevel_0+EducationField_Technical_DegreeXStockOptionLevel_0+TrainingTimesLastYearXYearsInCurrentRole+MonthlyIncomeXStockOptionLevel_1+TrainingTimesLastYearXYearsWithCurrManager+StockOptionLevel_0XWorkLifeBalance_Bad+YearsInCurrentRoleXDepartment_Research_AND_Development+MaritalStatus_SingleXPerformanceRating_Outstanding+Department_Research_AND_DevelopmentXJobLevel_2+DailyRateXMaritalStatus_Divorced+YearsInCurrentRoleXJobInvolvement_Low+RelationshipSatisfaction_LowXStockOptionLevel_0+JobInvolvement_HighXMaritalStatus_Divorced+TrainingTimesLastYearXYearsAtCompany+JobRole_Laboratory_TechnicianXStockOptionLevel_0+YearsAtCompany+YearsAtCompanyXDepartment_Research_AND_Development+DailyRateXYearsWithCurrManager+WorkLifeBalance_Bad+Gender_FemaleXOverTime_No+OverTime_NoXRelationshipSatisfaction_Medium+YearsAtCompanyXPerformanceRating_Excellent+MonthlyRateXDepartment_Research_AND_Development+JobSatisfaction_HighXStockOptionLevel_0+OverTime_YesXStockOptionLevel_3+TrainingTimesLastYearXDepartment_Research_AND_Development+EnvironmentSatisfaction_MediumXOverTime_No+MonthlyRateXJobLevel_2+Education_CollegeXRelationshipSatisfaction_Low+JobSatisfaction_Very_HighXStockOptionLevel_1+DailyRateXWorkLifeBalance_Bad'
formula_str <- 'Attrition ~ JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+DistanceFromHomeXStockOptionLevel_0+Department_SalesXJobLevel_1+DailyRateXStockOptionLevel_0+JobInvolvement_Low+OverTime_NoXStockOptionLevel_1+MonthlyIncomeXTrainingTimesLastYear+Age'
thresh <- .6

metric <- 0
num_passed <- 0
set.seed(seed) # re-setting the seed for some sort of predictability
for (jj in 1:num_tests){
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  model <- naiveBayes(as.formula(formula_str), data = train)
  pred <- predict(model, validation)
  cmBayes <- confusionMatrix(table(pred,validation$Attrition))
  sens <- as.numeric(cmBayes$byClass["Sensitivity"])
  if(is.na(sens)) {
    sens<-0
  }
  spec <- as.numeric(cmBayes$byClass["Specificity"])
  if(is.na(spec)){
    spec<-0
  }
  if(sens>thresh && spec>thresh) {
    num_passed <- num_passed + 1
  }
  metric <- metric + sens + spec
}
metric / num_tests
num_passed

# Determine how well 3 variables works for KNN
seed <- 10
prop_test <- .1
num_tests <- 200
vars <- c('JobLevel_1XOverTime_Yes','OverTime_YesXStockOptionLevel_0','TrainingTimesLastYearXOverTime_Yes','Education_BachelorXOverTime_Yes','JobRole_Sales_RepresentativeXPerformanceRating_Excellent','StockOptionLevel_1')
thresh <- .6
k_to_use <- 13

metric <- 0
num_passed <- 0
set.seed(seed) # re-setting the seed for some sort of predictability
for (jj in 1:num_tests){
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  classifications <- knn(train[,vars], # you might need at least 2 variables
                         validation[,vars],
                         train$Attrition, prob = TRUE, k = k_to_use)
  CM = confusionMatrix(table(validation$Attrition,classifications))
  sens <- as.numeric(CM$byClass["Sensitivity"])
  if(is.na(sens)) {
    sens<-0
  }
  spec <- as.numeric(CM$byClass["Specificity"])
  if(is.na(spec)){
    spec<-0
  }
  if(sens>thresh && spec>thresh) {
    num_passed <- num_passed + 1
  }
  metric <- metric + sens + spec
}
metric / num_tests
num_passed

# Play around with XGBoost
seed <- 1
prop_test <- .1
set.seed(seed)
indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
validation = data[indices,]
train = data[-indices,]

vals <- c('OverTime','MaritalStatus')
target <- as.numeric(train$Attrition)*-1 +2
feature_data <- train[,vals]
feature_matrix <- as.matrix(as.data.frame(lapply(feature_data,as.numeric)))
dtrain <- xgb.DMatrix(data = feature_matrix, label = target)
bstDMatrix <- xgboost(data = dtrain, nthread = 2, nrounds = 10, objective = "binary:logistic")
validation_matrix <- as.matrix(as.data.frame(lapply(validation[,vals],as.numeric)))
validation_dmatrix <- xgb.DMatrix(data = validation_matrix)
pred <- predict(bstDMatrix, validation_dmatrix)
prediction <- as.numeric(pred > 0.5)
validation_label <- as.numeric(validation[, 'Attrition'])*-1+2
sens <- sum(prediction == 1 & validation_label == 1) / sum(prediction == 1)
spec <- sum(prediction == 0 & validation_label == 0) / sum(prediction == 0)
sens + spec # It seems like it's doing well enough to give it a try using the approach I used before

# Try calling the function for Attrition
seed <- 1
prop_test <- .1
thresh <- .6
num_tests <- 200
num_no_change <- 50
start_formula <- 'JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0'
start_num <- 19
start_best <- 1.563
txt_file <- 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Save_XGBoost_Attrition.txt'
amount_to_improve <- 0.01
vars_df <- add_vars_xgboost_attrition(data, all_vars, seed, prop_test, thresh, num_tests, 
                                       num_no_change, start_formula, start_num, txt_file, 
                                       start_best, amount_to_improve)
write.xlsx(vars_df,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/vars_df.xlsx',colNames = TRUE)

# What happens with just the two variables?
seed <- 10
prop_test <- .1
num_tests <- 200
thresh <- .6
set.seed(seed)
num_passed <- 0
metric <- 0
for (ii in 1:num_tests){
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  
  vals <- c('OverTime','MaritalStatus')
  target <- as.numeric(train$Attrition)*-1 +2
  feature_data <- train[,vals]
  feature_matrix <- as.matrix(as.data.frame(lapply(feature_data,as.numeric)))
  dtrain <- xgb.DMatrix(data = feature_matrix, label = target)
  msg_output <- capture.output({
    bstDMatrix <- xgboost(data = dtrain, nthread = 2, nrounds = 10, objective = "binary:logistic")
  })
  validation_matrix <- as.matrix(as.data.frame(lapply(validation[,vals],as.numeric)))
  validation_dmatrix <- xgb.DMatrix(data = validation_matrix)
  pred <- predict(bstDMatrix, validation_dmatrix)
  prediction <- as.numeric(pred > 0.5)
  validation_label <- as.numeric(validation[, 'Attrition'])*-1+2
  sens <- sum(prediction == 1 & validation_label == 1) / sum(prediction == 1)
  if(is.na(sens)){
    sens <- 0
  }
  spec <- sum(prediction == 0 & validation_label == 0) / sum(prediction == 0)
  if(is.na(spec)){
    spec <- 0
  }
  if(sens > thresh && spec > thresh){
    num_passed <- num_passed + 1
  }
  metric <- metric + sens + spec
}
metric/num_tests
num_passed


# Re-trying NB with the 0.01 addition
seed <- 1
prop_test <- .1
thresh <- .6
num_tests <- 200
num_no_change <- 50
start_formula <- ''
start_num <- 1
start_best <- 0
txt_file <- 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Save_NB.txt'
amount_to_improve <- 0.01
vars_df <- add_vars_nb(data, all_vars, seed, prop_test, thresh, num_tests, 
                                      num_no_change, start_formula, start_num, txt_file, 
                                      start_best, amount_to_improve)
write.xlsx(vars_df,'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Excels/vars_df.xlsx',colNames = TRUE)
# Definitely didn't perform as well, but it has less variables

# Make predictions for Attrition
data2 <- read.table(textConnection(getURL(
  "https://s3.us-east-2.amazonaws.com/ddsproject1/CaseStudy2CompSet%20No%20Attrition.csv"
)), sep=",", header=TRUE)

data2$Education[data2$Education == 1] <- 'Below College'
data2$Education[data2$Education == 2] <- 'College'
data2$Education[data2$Education == 3] <- 'Bachelor'
data2$Education[data2$Education == 4] <- 'Master'
data2$Education[data2$Education == 5] <- 'Doctor'
data2$EnvironmentSatisfaction[data2$EnvironmentSatisfaction == 1] <- 'Low'
data2$EnvironmentSatisfaction[data2$EnvironmentSatisfaction == 2] <- 'Medium'
data2$EnvironmentSatisfaction[data2$EnvironmentSatisfaction == 3] <- 'High'
data2$EnvironmentSatisfaction[data2$EnvironmentSatisfaction == 4] <- 'Very High'
data2$JobInvolvement[data2$JobInvolvement == 1] <- 'Low'
data2$JobInvolvement[data2$JobInvolvement == 2] <- 'Medium'
data2$JobInvolvement[data2$JobInvolvement == 3] <- 'High'
data2$JobInvolvement[data2$JobInvolvement == 4] <- 'Very High'
data2$JobSatisfaction[data2$JobSatisfaction == 1] <- 'Low'
data2$JobSatisfaction[data2$JobSatisfaction == 2] <- 'Medium'
data2$JobSatisfaction[data2$JobSatisfaction == 3] <- 'High'
data2$JobSatisfaction[data2$JobSatisfaction == 4] <- 'Very High'
data2$PerformanceRating[data2$PerformanceRating == 1] <- 'Low'
data2$PerformanceRating[data2$PerformanceRating == 2] <- 'Good'
data2$PerformanceRating[data2$PerformanceRating == 3] <- 'Excellent'
data2$PerformanceRating[data2$PerformanceRating == 4] <- 'Outstanding'
data2$RelationshipSatisfaction[data2$RelationshipSatisfaction == 1] <- 'Low'
data2$RelationshipSatisfaction[data2$RelationshipSatisfaction == 2] <- 'Medium'
data2$RelationshipSatisfaction[data2$RelationshipSatisfaction == 3] <- 'High'
data2$RelationshipSatisfaction[data2$RelationshipSatisfaction == 4] <- 'Very High'
data2$WorkLifeBalance[data2$WorkLifeBalance == 1] <- 'Bad'
data2$WorkLifeBalance[data2$WorkLifeBalance == 2] <- 'Good'
data2$WorkLifeBalance[data2$WorkLifeBalance == 3] <- 'Better'
data2$WorkLifeBalance[data2$WorkLifeBalance == 4] <- 'Best'

# Set variables to factors
data2_original <- data2
#data2$Attrition <- factor(data2$Attrition,levels=c("Yes","No")) # Making "Yes" the true value for bayes
data2$BusinessTravel <- as.factor(data2$BusinessTravel)
data2$Department <- as.factor(data2$Department)
data2$EducationField <- as.factor(data2$EducationField)
data2$Gender <- as.factor(data2$Gender)
data2$JobRole <- as.factor(data2$JobRole)
data2$MaritalStatus <- as.factor(data2$MaritalStatus)
data2$OverTime <- as.factor(data2$OverTime)
data2$Education <- factor(data2$Education,levels=c("Below College","College","Bachelor","Master","Doctor"))
data2$EnvironmentSatisfaction <- factor(data2$EnvironmentSatisfaction,levels=c("Low","Medium","High","Very High"))
data2$JobInvolvement <- factor(data2$JobInvolvement,levels=c("Low","Medium","High","Very High"))
data2$JobLevel <- as.factor(data2$JobLevel)
data2$JobSatisfaction <- factor(data2$JobSatisfaction,levels=c("Low","Medium","High","Very High"))
data2$PerformanceRating <- factor(data2$PerformanceRating,levels=c("Low","Good","Excellent","Outstanding"))
data2$RelationshipSatisfaction <- factor(data2$RelationshipSatisfaction,levels=c("Low","Medium","High","Very High"))
data2$StockOptionLevel <- as.factor(data2$StockOptionLevel)
data2$WorkLifeBalance <- factor(data2$WorkLifeBalance,levels=c("Bad","Good","Better","Best"))

# Make a YearsAtCompany_Capped variable and cap it at 30
data2$YearsAtCompany_Capped <- pmin(data2$YearsAtCompany,30)

# Create lists of the variables
# List all the categorical and numeric variables
cat_vars <- c('BusinessTravel','Department','Education','EducationField',
              'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole',
              'JobSatisfaction','MaritalStatus','OverTime','PerformanceRating',
              'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance')
num_vars <- c('Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome',
              'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
              'TrainingTimesLastYear','YearsAtCompany','YearsAtCompany_Capped',
              'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager')

# Create square vars
nvars <- length(num_vars)
squared_vars <- c()
data2_no_squared <- data2
for (ii in 1:nvars){
  var <- num_vars[ii]
  var_squared <- paste(var,"_squared",sep="")
  data2[,var_squared]<-data2[,var]*data2[,var]
  squared_vars <- c(squared_vars,var_squared)
}

# Get onehot encoded variables
before_onehot <- data2
data2 <- one_hot(as.data.table(data2))
data2 <- as.data.frame(data2)
data2$Attrition <- before_onehot$Attrition
data2$Attrition_Yes <- c()
data2$Attrition_No <- c()
nvar <- length(cat_vars)
for (ii in 1:nvar){
  var <- cat_vars[ii]
  data2[,var] <- before_onehot[,var]
}
names(data2) <- gsub(' ','_',names(data2))
names(data2) <- gsub('&','AND',names(data2))
names(data2) <- gsub('-','_',names(data2))
col_names <- names(data2)
onehot_vars <- col_names[grepl('_', names(data2))]
onehot_vars <- onehot_vars[!grepl('squared', onehot_vars)]

# Make cross products of variables
data2_before_cross <- data2
to_cross_vars <- c(num_vars,onehot_vars)
nvars <- length(to_cross_vars)
cross_vars <- c()
for (ii in 1:nvars){
  for(jj in 1:nvars){
    if (ii<jj){
      var1 <- to_cross_vars[ii]
      var2 <- to_cross_vars[jj]
      new_col <- data2[,var1]*data2[,var2]
      if(sum(new_col)!=0){
        new_var <- paste(var1,var2,sep="X")
        data2[,new_var]<-new_col
        cross_vars <- c(cross_vars,new_var)
      }
    }
  }
}

# Smaller model
vars <- c('JobLevel_1XOverTime_Yes','OverTime_YesXStockOptionLevel_0','TrainingTimesLastYearXOverTime_Yes','Education_BachelorXOverTime_Yes','JobRole_Sales_RepresentativeXPerformanceRating_Excellent','StockOptionLevel_1')
k_to_use <- 13
classifications <- knn(data[,vars], # you might need at least 2 variables
                       data2[,vars],
                       data$Attrition, prob = TRUE, k = k_to_use) # 3 yes

write.xlsx(data2, 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Predictions/Attrition_Model_1.xlsx',colNames = TRUE)

# Larger model
formula_str <- 'Attrition~JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+JobRole_Sales_RepresentativeXStockOptionLevel_0+MaritalStatus+MonthlyRateXOverTime_Yes+PercentSalaryHikeXJobInvolvement_Low+YearsInCurrentRoleXOverTime_No+DailyRateXStockOptionLevel_1+OverTime_NoXStockOptionLevel_1+JobInvolvement_LowXPerformanceRating_Excellent+OverTime_NoXWorkLifeBalance_Better+JobLevel_1XPerformanceRating_Excellent+TotalWorkingYearsXTrainingTimesLastYear+PerformanceRating_ExcellentXStockOptionLevel_0+JobRole_Laboratory_TechnicianXWorkLifeBalance_Bad+TotalWorkingYearsXStockOptionLevel_1+YearsAtCompanyXOverTime_No+MonthlyIncomeXJobLevel_1+MonthlyIncomeXTrainingTimesLastYear+JobSatisfaction_LowXStockOptionLevel_0+EducationField_Technical_DegreeXStockOptionLevel_0+TrainingTimesLastYearXYearsInCurrentRole+MonthlyIncomeXStockOptionLevel_1+TrainingTimesLastYearXYearsWithCurrManager+StockOptionLevel_0XWorkLifeBalance_Bad+YearsInCurrentRoleXDepartment_Research_AND_Development+MaritalStatus_SingleXPerformanceRating_Outstanding+Department_Research_AND_DevelopmentXJobLevel_2+DailyRateXMaritalStatus_Divorced+YearsInCurrentRoleXJobInvolvement_Low+RelationshipSatisfaction_LowXStockOptionLevel_0+JobInvolvement_HighXMaritalStatus_Divorced+TrainingTimesLastYearXYearsAtCompany+JobRole_Laboratory_TechnicianXStockOptionLevel_0+YearsAtCompany+YearsAtCompanyXDepartment_Research_AND_Development+DailyRateXYearsWithCurrManager+WorkLifeBalance_Bad+Gender_FemaleXOverTime_No+OverTime_NoXRelationshipSatisfaction_Medium+YearsAtCompanyXPerformanceRating_Excellent+MonthlyRateXDepartment_Research_AND_Development+JobSatisfaction_HighXStockOptionLevel_0+OverTime_YesXStockOptionLevel_3+TrainingTimesLastYearXDepartment_Research_AND_Development+EnvironmentSatisfaction_MediumXOverTime_No+MonthlyRateXJobLevel_2+Education_CollegeXRelationshipSatisfaction_Low+JobSatisfaction_Very_HighXStockOptionLevel_1+DailyRateXWorkLifeBalance_Bad'
formula_obj <- as.formula(formula_str)
model <- naiveBayes(formula_obj, data = data)
pred <- predict(model, data2) # 84 yes
data2$Attrition <- pred
write.xlsx(data2, 'C:/Users/aabro/OneDrive/Desktop/SMU Program/Doing Data Science/Units/Unit 14 and 15 - Project 2 (Individual)/Predictions/Attrition_Model_2.xlsx',colNames = TRUE)

# I didn't realize the models were so different, how do they differ on the normal data?
vars <- c('JobLevel_1XOverTime_Yes','OverTime_YesXStockOptionLevel_0','TrainingTimesLastYearXOverTime_Yes','Education_BachelorXOverTime_Yes','JobRole_Sales_RepresentativeXPerformanceRating_Excellent','StockOptionLevel_1')
k_to_use <- 13
classifications <- knn(data[,vars], # you might need at least 2 variables
                       data[,vars],
                       data$Attrition, prob = TRUE, k = k_to_use) # 39 yes

formula_str <- 'Attrition~JobLevel_1XOverTime_Yes+OverTime_YesXStockOptionLevel_0+JobRole_Sales_RepresentativeXStockOptionLevel_0+MaritalStatus+MonthlyRateXOverTime_Yes+PercentSalaryHikeXJobInvolvement_Low+YearsInCurrentRoleXOverTime_No+DailyRateXStockOptionLevel_1+OverTime_NoXStockOptionLevel_1+JobInvolvement_LowXPerformanceRating_Excellent+OverTime_NoXWorkLifeBalance_Better+JobLevel_1XPerformanceRating_Excellent+TotalWorkingYearsXTrainingTimesLastYear+PerformanceRating_ExcellentXStockOptionLevel_0+JobRole_Laboratory_TechnicianXWorkLifeBalance_Bad+TotalWorkingYearsXStockOptionLevel_1+YearsAtCompanyXOverTime_No+MonthlyIncomeXJobLevel_1+MonthlyIncomeXTrainingTimesLastYear+JobSatisfaction_LowXStockOptionLevel_0+EducationField_Technical_DegreeXStockOptionLevel_0+TrainingTimesLastYearXYearsInCurrentRole+MonthlyIncomeXStockOptionLevel_1+TrainingTimesLastYearXYearsWithCurrManager+StockOptionLevel_0XWorkLifeBalance_Bad+YearsInCurrentRoleXDepartment_Research_AND_Development+MaritalStatus_SingleXPerformanceRating_Outstanding+Department_Research_AND_DevelopmentXJobLevel_2+DailyRateXMaritalStatus_Divorced+YearsInCurrentRoleXJobInvolvement_Low+RelationshipSatisfaction_LowXStockOptionLevel_0+JobInvolvement_HighXMaritalStatus_Divorced+TrainingTimesLastYearXYearsAtCompany+JobRole_Laboratory_TechnicianXStockOptionLevel_0+YearsAtCompany+YearsAtCompanyXDepartment_Research_AND_Development+DailyRateXYearsWithCurrManager+WorkLifeBalance_Bad+Gender_FemaleXOverTime_No+OverTime_NoXRelationshipSatisfaction_Medium+YearsAtCompanyXPerformanceRating_Excellent+MonthlyRateXDepartment_Research_AND_Development+JobSatisfaction_HighXStockOptionLevel_0+OverTime_YesXStockOptionLevel_3+TrainingTimesLastYearXDepartment_Research_AND_Development+EnvironmentSatisfaction_MediumXOverTime_No+MonthlyRateXJobLevel_2+Education_CollegeXRelationshipSatisfaction_Low+JobSatisfaction_Very_HighXStockOptionLevel_1+DailyRateXWorkLifeBalance_Bad'
formula_obj <- as.formula(formula_str)
model <- naiveBayes(formula_obj, data = data)
pred <- predict(model, data) # 225 yes

# I saw people changing the probability values of success.
# There's not enough time for me to look into that too deep.
# Determine how well 3 variables works for KNN
classifications <- knn(train[,vars], # you might need at least 2 variables
                       validation[,vars],
                       train$Attrition, prob = TRUE, k = k_to_use)
prob_level <- .55
probs = ifelse(classifications == "Yes",attributes(classifications)$prob, 1- attributes(classifications)$prob)
NewClass = ifelse(probs > prob_level, "Yes", "No")

seed <- 10
prop_test <- .1
num_tests <- 200
vars <- c('JobLevel_1XOverTime_Yes','OverTime_YesXStockOptionLevel_0','TrainingTimesLastYearXOverTime_Yes','Education_BachelorXOverTime_Yes','JobRole_Sales_RepresentativeXPerformanceRating_Excellent','StockOptionLevel_1')
thresh <- .6
k_to_use <- 13
prob_level <- .6

metric <- 0
num_passed <- 0
set.seed(seed) # re-setting the seed for some sort of predictability
for (jj in 1:num_tests){
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  classifications <- knn(train[,vars], # you might need at least 2 variables
                         validation[,vars],
                         train$Attrition, prob = TRUE, k = k_to_use)
  probs = ifelse(classifications == "Yes",attributes(classifications)$prob, 1- attributes(classifications)$prob)
  NewClass = ifelse(probs >= prob_level, "Yes", "No")
  CM = confusionMatrix(table(validation$Attrition,factor(NewClass, levels=c("Yes","No"))))
  sens <- as.numeric(CM$byClass["Sensitivity"])
  if(is.na(sens)) {
    sens<-0
  }
  spec <- as.numeric(CM$byClass["Specificity"])
  if(is.na(spec)){
    spec<-0
  }
  if(sens>thresh && spec>thresh) {
    num_passed <- num_passed + 1
  }
  metric <- metric + sens + spec
}
metric / num_tests
num_passed
# .5 and .55 are the best, but not too different.  I'll just stick with .5