# This function loops through a bunch of variables and creates a naive bayes model with each.
# It then calculates the statistics like accuracy, sensitivity, specificity, etc.
bayes_accuracies <- function(df, dep_var, indep_vars){
  numVars <- nrow(indep_vars)
  for (ii in 1:numVars){
    var = indep_vars$Variable[ii]
    nb_model <- naiveBayes(!!sym(dep_var) ~ !!sym(var), data = data)
    preds <- predict(nb_model, newdata=data)
    cmBayes <- confusionMatrix(table(preds,data$Attrition))
  }
}

# This function loops through all the variables, removes the worst, then gets the sensitivy + specificity
prune_vars_nb <- function(data, all_vars, seed, prop_test, num_ensemble){
  set.seed(seed)
  num_vars <- length(all_vars)
  nb_vars <- data.frame(num_vars = (seq(0:num_vars)-1))
  nb_vars$var_removed <- ""
  nb_vars$metric <- 0
  indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
  validation = data[indices,]
  train = data[-indices,]
  formula_str <- paste("Attrition ~ ", paste(all_vars,collapse="+"))
  # print(paste('formula_str: ', formula_str))
  formula_obj <- as.formula(formula_str)
  nb_models <- vector("list", num_ensemble)
  for (kk in 1:num_ensemble) {
    indices = sample(seq(1:nrow(train)),round(prop_test*nrow(train)))
    test <- train[indices,]
    train2 <- train[-indices,]
    nb_models[[kk]] <- naiveBayes(formula_obj, data = train2)
  }
  predictions <- sapply(nb_models, function(model) predict(model, validation))
  final_predictions <- apply(predictions, 1, function(x) {
    # Use majority vote to determine the final prediction
    tab <- table(x)
    factor(names(tab)[which.max(tab)],levels=c("Yes","No")) # Assuming factors where "Yes" is 1 and "No" is 0
  })
  cmBayes <- confusionMatrix(table(final_predictions,validation$Attrition))
  metric <- cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
  nb_vars$metric[num_vars+1] <- metric
  # print('Checked all variables')
  
  # After figuring out the metric with all of the variables, try it while removing each variable
  remaining_vars <- all_vars
  for(ii in (num_vars-1):1){ # Can't really remove the last variable
    num_remaining_vars <- length(remaining_vars)
    worst_var <- ""
    worst_metric <- 3
    for(jj in 1:num_remaining_vars){
      indices = sample(seq(1:nrow(train)),round(prop_test*nrow(train)))
      test = train[indices,]
      train2 = train[-indices,]
      var_to_remove <- remaining_vars[jj]
      vars_to_try <- remaining_vars[ !remaining_vars == var_to_remove]
      # print(paste(vars_to_try,collapse="+"))
      formula_str <- paste("Attrition ~ ", paste(vars_to_try,collapse="+"))
      formula_obj <- as.formula(formula_str)
      # print(paste('formula_str: ',formula_str))
      nb_models <- vector("list", num_ensemble)
      for (kk in 1:num_ensemble) {
        indices = sample(seq(1:nrow(train2)),round(prop_test*nrow(train2)))
        train3 <- train2[-indices,]
        nb_models[[kk]] <- naiveBayes(Attrition ~ ., data = train3)
      }
      predictions <- sapply(nb_models, function(model) predict(model, test))
      final_predictions <- apply(predictions, 1, function(x) {
        # Use majority vote to determine the final prediction
        tab <- table(x)
        factor(names(tab)[which.max(tab)],levels=c("Yes","No")) # Assuming factors where "Yes" is 1 and "No" is 0
      })
      cmBayes <- confusionMatrix(table(final_predictions,test$Attrition))
      metric <- cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
      if(metric < worst_metric){
        worst_var <- var_to_remove
        worst_metric <- metric
      }
    }
    
    # After you've figure out the worst metric, see what removing it does to the score
    remaining_vars <- remaining_vars[ !remaining_vars == worst_var]
    formula_str <- paste("Attrition ~ ", paste(remaining_vars,collapse="+"))
    formula_obj <- as.formula(formula_str)
    nb_models <- vector("list", num_ensemble)
    for (kk in 1:num_ensemble) {
      indices = sample(seq(1:nrow(train)),round(prop_test*nrow(train)))
      test <- train[indices,]
      train2 <- train[-indices,]
      nb_models[[kk]] <- naiveBayes(formula_obj, data = train2)
    }
    predictions <- sapply(nb_models, function(model) predict(model, validation))
    final_predictions <- apply(predictions, 1, function(x) {
      # Use majority vote to determine the final prediction
      tab <- table(x)
      factor(names(tab)[which.max(tab)],levels=c("Yes","No")) # Assuming factors where "Yes" is 1 and "No" is 0
    })
    cmBayes <- confusionMatrix(table(final_predictions,validation$Attrition))
    metric <- cmBayes$byClass["Sensitivity"] + cmBayes$byClass["Specificity"]
    nb_vars$metric[num_remaining_vars] <- metric
    nb_vars$var_removed[num_remaining_vars] <- worst_var
    print(paste('Removed ', worst_var))
  }
  
  # Last remaining variable
  nb_vars$var_removed[1] <- remaining_vars
  
  # function output
  nb_vars
}

# This function goes through all of the variables in the p-value order
add_vars_nb <- function(data, all_vars, seed, prop_test, thresh, num_tests, 
                        num_no_change, start_formula, start_num, txt_file, 
                        start_best, amount_to_improve){
  # First get a list of the p-values
  var_df <- data.frame(Variables=all_vars)
  var_df$p <- 0
  var_df$metric <- 0
  var_df$num_passed <- 0
  var_df$to_use <- 0
  var_df$curr_formula <- ""
  num_since_last_change <- 0
  nvar <- length(all_vars)
  for(ii in 1:nvar){
    var<-all_vars[ii]
    test <- cor.test(unclass(data[,'Attrition']),unclass(data[,var]))
    var_df$p[ii] <- test$p.value
  }
  
  # Sort the p-values
  sort_order <- order(var_df$p,decreasing = FALSE)
  
  # Loop through all of the columns, add the next variable to the list
  if (start_formula==''){
    vars_to_use <- c()
  } else {
    vars_to_use <- unlist(strsplit(start_formula, "\\+"))
  }
  best_passed <- 0
  best_metric <- 0
  for (ii in 1:nvar){
    var<-all_vars[sort_order[ii]]
    vars_to_use_loop <- c(vars_to_use,var)
    curr_p <- var_df$p[sort_order[ii]]
    if(curr_p > 0.001){ # Don't want the variables to not be super predictive
      break
    }
    if(sum(data[,var]!=0)<length(data[,var])*0.05){
      next # Don't want too few numbers
    }
    formula_str <- paste("Attrition ~ ", paste(vars_to_use_loop,collapse="+"))
    formula_obj <- as.formula(formula_str)
    
    metric <- 0
    num_passed <- 0
    set.seed(seed) # re-setting the seed for some sort of predictability
    for (jj in 1:num_tests){
      indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
      validation = data[indices,]
      train = data[-indices,]
      model <- naiveBayes(formula_obj, data = train)
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
    metric <- metric / num_tests
    var_df$metric[var_df$Variable==var] <- metric
    var_df$num_passed[var_df$Variable==var] <- num_passed
    var_df$num[var_df$Variable==var] <- ii
    
    # Check if that variable should be added
    #if(num_passed>best_passed || num_passed<=num_tests*.01){
    if(best_metric+amount_to_improve<metric || num_passed<=num_tests*.01){
      best_passed = num_passed
      best_metric <- metric
      vars_to_use <- c(vars_to_use,var)
      var_df$to_use[var_df$Variable==var] <- 1
      var_df$curr_formula[var_df$Variable==var] <- formula_str
      num_since_last_change <- 0
      file_str <- paste(readLines(txt_file), collapse="\n")
      writeLines(paste(file_str,"\n",'num: ',ii,' var: ',var, 
                       ' metric: ', metric, ' num_passed: ', num_passed, 
                       ' formula_str: ',paste(vars_to_use,collapse="+"),sep=""),
                 txt_file)
    }
    
    # Print something to the screen to see general progress
    print(paste('num: ',ii,' var: ',var,' metric: ', metric,
                ' num_since_last_change: ',num_since_last_change,
                ' num_passed: ', num_passed, sep=""))
    
    # If the loop has been going on for too long, just break out of it
    num_since_last_change <- num_since_last_change + 1
    if(num_since_last_change == num_no_change){
      break
    }
  }
  
  # Output the var_df
  var_df
}

# This function goes through all of the variables in the p-value order
add_vars_reg <- function(data, all_vars, seed, prop_test, thresh, num_tests, 
                         num_no_change, start_formula, start_num, txt_file, 
                         start_best, amount_to_improve, var_to_predict){
  # First get a list of the p-values
  var_df <- data.frame(Variables=all_vars)
  var_df$num <- 0
  var_df$p <- 0
  var_df$metric <- 0
  var_df$num_passed <- 0
  var_df$to_use <- 0
  var_df$curr_formula <- ""
  nvar <- length(all_vars)
  for(ii in 1:nvar){
    var<-all_vars[ii]
    test <- cor.test(unclass(data[,var_to_predict]),unclass(data[,var]))
    var_df$p[ii] <- test$p.value
  }
  var_df$p[is.na(var_df$p)] <- 1
  
  # Sort the p-values
  sort_order <- order(var_df$p,decreasing = FALSE)
  
  # Loop through all of the columns, add the next variable to the list
  if (start_formula=='') {
    vars_to_use <- c()
  } else {
    vars_to_use <- strsplit(start_formula, "~")[[1]][2]
  }
  best_passed <- 0
  best_metric <- start_best
  for (ii in start_num:nvar){
    var<-all_vars[sort_order[ii]]
    curr_p <- var_df$p[sort_order[ii]]
    if(curr_p > 0.001){ # Don't want the variables to not be super predictive
      break
    }
    if(sum(data[,var]!=0)<length(data[,var])*0.05){
      next # Don't want too few numbers
    }
    
    vars_to_use_loop <- c(vars_to_use,var)
    formula_str <- paste(var_to_predict," ~ ", paste(vars_to_use_loop,collapse="+"))
    formula_obj <- as.formula(formula_str)
    
    metric <- 0
    num_passed <- 0
    set.seed(seed) # re-setting the seed for some sort of predictability
    for (jj in 1:num_tests){
      indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
      validation = data[indices,]
      train = data[-indices,]
      
      # Average of serveral models seems hard, so I'll just use one model
      model <- lm(formula_obj, data = train)
      
      # Test model
      pred <- predict(model, validation)
      err <- sqrt(sum((pred-as.numeric(validation[,var_to_predict]))^2)/nrow(validation))
      if(err<thresh) {
        num_passed <- num_passed + 1
      }
      metric <- metric + err
      
    }
    metric <- metric / num_tests
    var_df$metric[var_df$Variable==var] <- metric
    var_df$num_passed[var_df$Variable==var] <- num_passed
    
    # Check if that variable should be added
    if(best_metric>metric+amount_to_improve){
      best_passed = num_passed
      best_metric <- metric
      vars_to_use <- c(vars_to_use,var)
      var_df$to_use[var_df$Variable==var] <- 1
      var_df$curr_formula[var_df$Variable==var] <- formula_str
      var_df$num[var_df$Variable==var] <- ii
      num_since_last_change <- 0
      file_str <- paste(readLines(txt_file), collapse="\n")
      writeLines(paste(file_str,"\n",'num: ',ii,' var: ',var, 
                       ' metric: ', metric, ' num_passed: ', num_passed, 
                       ' formula_str: ',paste(vars_to_use,collapse="+"),sep=""),
                 txt_file)
    }
    
    # Print something to the screen to see general progress
    print(paste('num: ',ii,' var: ',var,' metric: ', metric,
                ' num_since_last_change: ',num_since_last_change,
                ' num_passed: ', num_passed, sep=""))
    
    # If the loop has been going on for too long, or if p-values aren't low anymore, just break out of it
    num_since_last_change <- num_since_last_change + 1
    if(num_since_last_change == num_no_change){
      break
    }
  }
  
  # Output the var_df
  var_df
}

add_vars_knn <- function(data, all_vars, seed, prop_test, thresh, num_tests, num_no_change, start_formula, start_num, txt_file, start_best){
  # First get a list of the p-values
  var_df <- data.frame(Variables=all_vars)
  var_df$num <- 0
  var_df$p <- 0
  var_df$metric <- 0
  var_df$num_passed <- 0
  var_df$k <- 0
  var_df$to_use <- 0
  var_df$curr_formula <- ""
  num_since_last_change <- 0
  nvar <- length(all_vars)
  for(ii in 1:nvar){
    var<-all_vars[ii]
    # print(var)
    test <- cor.test(unclass(data[,'Attrition']),unclass(data[,var]))
    var_df$p[ii] <- test$p.value
  }
  
  # Sort the p-values
  sort_order <- order(var_df$p,decreasing = FALSE)
  
  # Loop through all of the columns, add the next variable to the list
  if (start_formula==''){
    vars_to_use <- c()
  } else {
    vars_to_use <- unlist(strsplit(start_formula, "\\+"))
  }
  best_passed <- 0
  best_metric <- start_best
  for (ii in start_num:nvar){
    var<-all_vars[sort_order[ii]]
    vars_to_use_loop <- c(vars_to_use,var)
    curr_p <- var_df$p[sort_order[ii]]
    if(curr_p > 0.001){ # Don't want the variables to not be super predictive
      break
    }
    if(sum(data[,var]!=0)<length(data[,var])*0.05){
      next # Don't want too few numbers
    }
    if(class(data[,var]) == 'factor'){
      next # KNN only really handles numbers
    }
    
    k_met <- 0
    k_num_passed <- 0
    k_best <- 3
    set.seed(seed) # re-setting the seed for some sort of predictability
    for (kk in c(3,5,7,9,11,13,15)){
    # for (kk in c(3,5)){
      print(paste('k is ',kk,sep=""))
      metric <- 0
      num_passed <- 0
      for (jj in 1:num_tests){
        indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
        validation = data[indices,]
        train = data[-indices,]
        classifications <- knn(train[,vars_to_use_loop], # you might need at least 2 variables
                               validation[,vars_to_use_loop],
                               train$Attrition, prob = TRUE, k = kk)
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
      
      metric <- metric / num_tests
      if(k_met<metric){
        k_best <- kk
        k_met <- metric
        k_num_passed <- num_passed
      }
    }
    var_df$metric[var_df$Variable==var] <- k_met
    var_df$num_passed[var_df$Variable==var] <- k_num_passed
    var_df$num[var_df$Variable==var] <- ii
    
    # Check if that variable should be added
    if(best_metric+0.01<k_met || k_num_passed<=num_tests*.01){
      best_passed = k_num_passed
      best_metric <- k_met
      vars_to_use <- c(vars_to_use,var)
      var_df$to_use[var_df$Variable==var] <- 1
      var_df$curr_formula[var_df$Variable==var] <- paste(vars_to_use,collapse="+")
      var_df$k[var_df$Variable==var] <- k_best
      num_since_last_change <- 0
      file_str <- paste(readLines(txt_file), collapse="\n")
      writeLines(paste(file_str,"\n",'num: ',ii,' var: ',var, ' k: ', k_best,
                       ' metric: ', metric, ' num_passed: ', num_passed, 
                       ' formula_str: ',paste(vars_to_use,collapse="+"),sep=""),
                 txt_file)
    }
    
    # Print something to the screen to see general progress
    print(paste('num: ',ii,' var: ',var,' metric: ', metric,
                ' num_since_last_change: ',num_since_last_change,
                ' num_passed: ', num_passed, " k: ", k_best, sep=""))
    
    # If the loop has been going on for too long, just break out of it
    num_since_last_change <- num_since_last_change + 1
    if(num_since_last_change == num_no_change){
      break
    }
  }
  
  # Output the var_df
  var_df
}

add_vars_xgboost_attrition <- function(data, all_vars, seed, prop_test, thresh, num_tests, 
                                       num_no_change, start_formula, start_num, txt_file, 
                                       start_best, amount_to_improve){
  # First get a list of the p-values
  var_df <- data.frame(Variables=all_vars)
  var_df$num <- 0
  var_df$p <- 0
  var_df$metric <- 0
  var_df$num_passed <- 0
  var_df$to_use <- 0
  var_df$curr_formula <- ""
  num_since_last_change <- 0
  nvar <- length(all_vars)
  for(ii in 1:nvar){
    var<-all_vars[ii]
    test <- cor.test(unclass(data[,'Attrition']),unclass(data[,var]))
    var_df$p[ii] <- test$p.value
  }
  
  # Sort the p-values
  sort_order <- order(var_df$p,decreasing = FALSE)
  
  # Loop through all of the columns, add the next variable to the list
  if (start_formula==''){
    vars_to_use <- c()
  } else {
    vars_to_use <- unlist(strsplit(start_formula, "\\+"))
  }
  best_passed <- 0
  best_metric <- start_best
  for (ii in start_num:nvar){
    var<-all_vars[sort_order[ii]]
    vars_to_use_loop <- c(vars_to_use,var)
    curr_p <- var_df$p[sort_order[ii]]
    if(curr_p > 0.001){ # Don't want the variables to not be super predictive
      break
    }
    if(sum(data[,var]!=0)<length(data[,var])*0.05){
      next # Don't want too few numbers
    }
    if(class(data[,var]) == 'factor'){
      next # XGBoost only seems to handle numeric data well
    }
    
    metric <- 0
    num_passed <- 0
    set.seed(seed) # re-setting the seed for some sort of predictability
    for (jj in 1:num_tests){
      # Split data into training and test
      indices = sample(seq(1:nrow(data)),round(prop_test*nrow(data)))
      validation = data[indices,]
      train = data[-indices,]
      
      # Call the xgboost algorithm
      target <- as.numeric(train$Attrition)*-1 +2
      feature_data <- train[,vars_to_use_loop]
      feature_matrix <- as.matrix(feature_data)
      dtrain <- xgb.DMatrix(data = feature_matrix, label = target)
      msg_output <- capture.output({
        bstDMatrix <- xgboost(data = dtrain, nthread = 2, nrounds = 10, objective = "binary:logistic")
      })
        
      # Use the resulting model for predictions
      validation_matrix <- as.matrix(validation[,vars_to_use_loop])
      validation_dmatrix <- xgb.DMatrix(data = validation_matrix)
      pred <- predict(bstDMatrix, validation_dmatrix)
      prediction <- as.numeric(pred > 0.5)
      
      # Determine how good the predictions are
      validation_label <- as.numeric(validation[, 'Attrition'])*-1+2
      sens <- sum(prediction == 1 & validation_label == 1) / sum(prediction == 1)
      if(is.na(sens)){
        sens <- 0
      }
      spec <- sum(prediction == 0 & validation_label == 0) / sum(prediction == 0)
      if(is.na(spec)){
        spec <- 0
      }
      if(sens>thresh && spec>thresh) {
        num_passed <- num_passed + 1
      }
      metric <- metric + sens + spec
      
    }
    metric <- metric / num_tests
    var_df$metric[var_df$Variable==var] <- metric
    var_df$num_passed[var_df$Variable==var] <- num_passed
    var_df$num[var_df$Variable==var] <- ii
    
    # Check if that variable should be added
    if(best_metric+amount_to_improve<metric || num_passed<=num_tests*.01){
      best_passed = num_passed
      best_metric <- metric
      vars_to_use <- c(vars_to_use,var)
      var_df$to_use[var_df$Variable==var] <- 1
      var_df$curr_formula[var_df$Variable==var] <- paste(vars_to_use,collapse="+")
      num_since_last_change <- 0
      file_str <- paste(readLines(txt_file), collapse="\n")
      writeLines(paste(file_str,"\n",'num: ',ii,' var: ',var, 
                       ' metric: ', metric, ' num_passed: ', num_passed, 
                       ' formula_str: ',paste(vars_to_use,collapse="+"),sep=""),
                 txt_file)
    }
    
    # Print something to the screen to see general progress
    print(paste('num: ',ii,' var: ',var,' metric: ', metric,
                ' num_since_last_change: ',num_since_last_change,
                ' num_passed: ', num_passed, sep=""))
    
    # If the loop has been going on for too long, just break out of it
    num_since_last_change <- num_since_last_change + 1
    if(num_since_last_change == num_no_change){
      break
    }
  }
  
  # Output the var_df
  var_df
}