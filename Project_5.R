#Section 1 : In this section, we first install all the necessary libraries, import them, set the tasks and define configurations related to the AutoTuner class

install.packages(c('mlr3','mlr3verse', 'mlr3learners', 'paradox', 'mlr3tuning', 'mlr3pipelines', 'mlr3viz', 'mlr3tuningspaces', 'mlr3oml', 'ggplot2', 'dplyr', 'gridExtra', 'igraph', 'ranger', 'xgboost', 'R6'))

library(mlr3)
library(mlr3verse)
library(mlr3learners)
library(paradox)
library(mlr3tuning)  
library(mlr3pipelines)
library(mlr3viz)
library(mlr3tuningspaces)
library(mlr3oml)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(igraph)
library(ranger)
library(xgboost)
library(R6)

#We first set a seed to ensure reproducibility 
set.seed(1)


tasks = list( tsk('pima') , tsk('sonar'), tsk("oml", task_id = 10093), tsk('oml', task_id=10101) , tsk('oml', task_id= 15)) 
# Tasks breakdown: 
# Task:             'pima'      -       Pima Indians Diabetes Dataset              https://www.openml.org/search?type=data&status=active&id=37
# Task:             'sonar'     -       Sonar Dataset                              https://www.openml.org/search?type=data&status=active&id=40
# Task with ID:     '10093'     -       Banknote Authentication Dataset            https://www.openml.org/search?type=data&status=active&id=1462
# Task with ID:     '10101'     -       Blood Transfusion Service Center Dataset   https://www.openml.org/search?type=data&status=active&id=1464
# Task with ID:     '15'        -       Breast Cancer Wisconsin Dataset            https://www.openml.org/search?type=data&status=active&id=15

rand_tuner = tnr("random_search")         # Defining that the Tuner should be Random Search Tuner | Can be - Bayesian Optimization (mbo), Grid Search (grid_search) etc
terminator = trm("run_time", secs = 15)   # Termination Criteria : Hyperparamter Search ends after 15 seconds, can be decreased to reduce the searching time but may affect classfication accuracy
rsmp_tuner = rsmp("cv", folds = 3)        # Resampling that is used to evaluate the performance of the hyperparameter configurations



# Section 2: In this section, we define the learners (classifiers), the hyperparameter search space for different learners and benchmark the performance of the learners on the above tasks (datasets) with respect to different evaluation measures
# Using mlr3pipelines, learner pipelines are created that include mean imputation and scaling preprocessing operators (PipeOps) 


# Baseline Classifier - A featureless classifier which makes random guesses for the correct class
featureless_lrn = lrn('classif.featureless', predict_type = 'prob')
featureless_lrn = po('imputemean') %>>%  po('scale')  %>>%  featureless_lrn         # Joining the featureless learner with 2 preprocessing operators using %>>% (PipeOps join operator)
plot(featureless_lrn, html=FALSE)

# Random Forest Classfier with defined parameter search space (mtry.ratio and num.trees chosen to be tuned)
ranger_lrn = lrn("classif.ranger", mtry.ratio = to_tune(0.5,1), num.trees = to_tune(100,128), predict_type = "prob")
ranger_lrn = po('imputemean') %>>%  po('scale')  %>>%  ranger_lrn                   # Pre-processing pipeline with MeanImputation -> Feature Scaling -> Learner ( Random Forest Classifier )
plot(ranger_lrn, html=FALSE)

# Decision Tree Classfier with defined parameter search space (cp, minsplit and maxdepth chosen to be tuned) 
rpart_lrn = lrn("classif.rpart", cp = to_tune(1e-03, 1e-1, logscale = TRUE), minsplit = to_tune(64, 128), maxdepth = to_tune(10, 30), predict_type = "prob")
rpart_lrn = po('imputemean') %>>%  po('scale')  %>>%  rpart_lrn
plot(rpart_lrn, html=FALSE)

# XGBoost Classifier with defined parameter search space (nrounds, eta and maxdepth chosen to be tuned) 
xgboost_lrn = lrn("classif.xgboost", nrounds  = to_tune(50, 500), eta = to_tune(0.01, 0.3), max_depth = to_tune(1, 10), predict_type = "prob")
xgboost_lrn = po('imputemean') %>>%  po('scale')  %>>%  xgboost_lrn
plot(xgboost_lrn, html=FALSE)

# After defining the preprocessing pipelines, learners and the parameter space to be tuned on, now we must create AutoTuner class instances to set up Hyperparameter Optimization
# Defining the AutoTuner instance for Random Forest
ranger_auto_tuner = AutoTuner$new(      
  learner      = ranger_lrn,
  resampling   = rsmp_tuner, 
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,  
  terminator   = terminator
)
 
# Defining the AutoTuner instance for Decision Tree
rpart_auto_tuner = AutoTuner$new(
  learner      = rpart_lrn,
  resampling   = rsmp_tuner,
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,
  terminator   = terminator
)

# Defining the AutoTuner instance for XGBoost
xgboost_auto_tuner = AutoTuner$new(
  learner      = xgboost_lrn,
  resampling   = rsmp_tuner,
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,
  terminator   = terminator
)

# Benchmarking and Ranking Learners: Now all the learners aggregated with preprocessing pipelines will be benchmarked on the defined tasks and their ranks will be compared

learners = list(ranger_auto_tuner, rpart_auto_tuner, xgboost_auto_tuner, featureless_lrn)

# Creating a Benchmark Grid and Benchmarking: 
bm_design_1 = benchmark_grid(tasks, learners, rsmp("cv", folds=3)) # Creating a benchmarking design
bench_1     = benchmark(bm_design_1, store_models = TRUE) # Benchmarking operation: Takes time to execute depending on how huge Hyperparamter Search Space is
print(bench_1) # Prints if there were any errors or warnings during benchmarking process
autoplot(bench_1, type = "boxplot") # Boxplot for the classification scores for the different learners

# Ranking the learners on all tasks with different evaluation measures like Classification Error, Brier Score, AUC Score
tab_1   = bench_1$aggregate(measures = msrs(c("classif.ce", "classif.bbrier", "classif.auc"))) 
ranks_1 = tab_1[, .(learner_id , rankce = rank(classif.ce),rankbrier = rank(classif.bbrier), rankauc=rank(-classif.auc), classif.ce, classif.bbrier, classif.auc), by = task_id]
print(ranks_1)




# Section 3: We will now change the preprocessing pipeline by using Histogram Imputation instead of Mean Imputation and then benchmark the learners to see if there's a change in performance if the preprocessing pipeline is changed
# All other configurations such as learners, parameter search space, evaluation measures etc are kept constant 


# Baseline Classifier
featureless_lrn = lrn('classif.featureless', predict_type = 'prob')
featureless_lrn = po('imputehist') %>>%  po('scale')  %>>%  featureless_lrn         # Joining the featureless learner with 2 preprocessing operators using %>>% (PipeOps join operator)

# Random Forest Classfier 
ranger_lrn = lrn("classif.ranger", mtry.ratio = to_tune(0.5,1), num.trees = to_tune(100,128), predict_type = "prob")
ranger_lrn = po('imputehist') %>>%  po('scale')  %>>%  ranger_lrn                   # Pre-processing pipeline with HistImputation -> Feature Scaling -> Learner ( Random Forest Classifier )

# Decision Tree Classfier  
rpart_lrn = lrn("classif.rpart", cp = to_tune(1e-03, 1e-1, logscale = TRUE), minsplit = to_tune(64, 128), maxdepth = to_tune(10, 30), predict_type = "prob")
rpart_lrn = po('imputehist') %>>%  po('scale')  %>>%  rpart_lrn

# XGBoost Classifier 
xgboost_lrn = lrn("classif.xgboost", nrounds  = to_tune(50, 500), eta = to_tune(0.01, 0.3), max_depth = to_tune(1, 10), predict_type = "prob")
xgboost_lrn = po('imputehist') %>>%  po('scale')  %>>%  xgboost_lrn

ranger_auto_tuner = AutoTuner$new(      
  learner      = ranger_lrn,
  resampling   = rsmp_tuner, 
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,  
  terminator   = terminator
)
 rpart_auto_tuner = AutoTuner$new(
  learner      = rpart_lrn,
  resampling   = rsmp_tuner,
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,
  terminator   = terminator
)
xgboost_auto_tuner = AutoTuner$new(
  learner      = xgboost_lrn,
  resampling   = rsmp_tuner,
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,
  terminator   = terminator
)

# Benchmarking and Ranking Learners: Now all the learners aggregated with the changed preprocessing pipeline will be benchmarked on the defined tasks and their ranks will be compared again

learners = list(ranger_auto_tuner, rpart_auto_tuner, xgboost_auto_tuner, featureless_lrn)

# Creating a Benchmark Grid and Benchmarking: 
bm_design_2 = benchmark_grid(tasks, learners, rsmp("cv", folds=3)) #Creating a benchmarking design
bench_2     = benchmark(bm_design_2, store_models = TRUE) 
print(bench_2) 
autoplot(bench_2, type = "boxplot") 

# Ranking the learners again
tab_2   = bench_2$aggregate(measures = msrs(c("classif.ce", "classif.bbrier", "classif.auc"))) 
ranks_2 = tab_2[, .(learner_id , rankce = rank(classif.ce),rankbrier = rank(classif.bbrier), rankauc=rank(-classif.auc), classif.ce, classif.bbrier, classif.auc), by = task_id]
print(ranks_2)





# Section 4: We will change the preprocessing pipeline again by using a Variance Filter that removes features that mostly have noise (low variance) and combine it with Mean Imputation and Scaling PipeOps

# Baseline Classifier
featureless_lrn = lrn('classif.featureless', predict_type = 'prob')
featureless_lrn =  po('imputemean') %>>% po("filter", mlr3filters::flt("variance"), filter.frac = 0.5) %>>%   po('scale')  %>>%  featureless_lrn         # Joining the featureless learner with 2 preprocessing operators using %>>% (PipeOps join operator)

# Random Forest Classfier 
ranger_lrn = lrn("classif.ranger", mtry.ratio = to_tune(0.5,1), num.trees = to_tune(100,128), predict_type = "prob")
ranger_lrn = po('imputemean') %>>% po("filter", mlr3filters::flt("variance"), filter.frac = 0.5) %>>%   po('scale')  %>>%  ranger_lrn                   # Pre-processing pipeline with HistImputation -> Feature Scaling -> Learner ( Random Forest Classifier )

# Decision Tree Classfier  
rpart_lrn = lrn("classif.rpart", cp = to_tune(1e-03, 1e-1, logscale = TRUE), minsplit = to_tune(64, 128), maxdepth = to_tune(10, 30), predict_type = "prob")
rpart_lrn = po('imputemean') %>>% po("filter", mlr3filters::flt("variance"), filter.frac = 0.5) %>>%   po('scale')  %>>%  rpart_lrn

# XGBoost Classifier 
xgboost_lrn = lrn("classif.xgboost", nrounds  = to_tune(50, 500), eta = to_tune(0.01, 0.3), max_depth = to_tune(1, 10), predict_type = "prob")
xgboost_lrn = po('imputemean') %>>% po("filter", mlr3filters::flt("variance"), filter.frac = 0.5) %>>%   po('scale')  %>>%  xgboost_lrn

# AutoTuner Instances for each Learner
ranger_auto_tuner = AutoTuner$new(      
  learner      = ranger_lrn,
  resampling   = rsmp_tuner, 
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,  
  terminator   = terminator
)
 rpart_auto_tuner = AutoTuner$new(
  learner      = rpart_lrn,
  resampling   = rsmp_tuner,
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,
  terminator   = terminator
)
xgboost_auto_tuner = AutoTuner$new(
  learner      = xgboost_lrn,
  resampling   = rsmp_tuner,
  measure      = msr("classif.ce"),
  tuner        = rand_tuner,
  terminator   = terminator
)

# Benchmarking and Ranking Learners: Now all the learners aggregated with the changed preprocessing pipeline will be benchmarked on the defined tasks and their ranks will be compared again

learners = list(ranger_auto_tuner, rpart_auto_tuner, xgboost_auto_tuner, featureless_lrn)

# Creating a Benchmark Grid and Benchmarking: 
bm_design_3 = benchmark_grid(tasks, learners, rsmp("cv", folds=3)) #Creating a benchmarking design
bench_3     = benchmark(bm_design_3, store_models = TRUE) 
print(bench_3) 
autoplot(bench_3, type = "boxplot") 

# Ranking the learners again
tab_3   = bench_3$aggregate(measures = msrs(c("classif.ce", "classif.bbrier", "classif.auc"))) 
ranks_3 = tab_3[, .(learner_id , rankce = rank(classif.ce),rankbrier = rank(classif.bbrier), rankauc=rank(-classif.auc), classif.ce, classif.bbrier, classif.auc), by = task_id]
print(ranks_3)





# Section 5: We do not tune the Hyperparamters using the AutoTuner Class and set them to arbitrary values to see if there is a change in performance, preprocessing pipeline is kept same as the one used in Section 2

# Baseline Classifier
featureless_lrn = lrn('classif.featureless', predict_type = 'prob')
featureless_lrn = po('imputemean') %>>%  po('scale')  %>>%  featureless_lrn 

# Random Forest Classfier 
ranger_lrn = lrn("classif.ranger", mtry.ratio= 0.1, num.trees= 1, predict_type = "prob")
ranger_lrn = po('imputemean') %>>%  po('scale')  %>>%  ranger_lrn               

# Decision Tree Classfier 
rpart_lrn = lrn("classif.rpart", cp = 0.05, minsplit= 3, maxdepth=3, predict_type = "prob")
rpart_lrn = po('imputemean') %>>%  po('scale')  %>>%  rpart_lrn

# XGBoost Classifier 
xgboost_lrn= lrn("classif.xgboost",nrounds  = 25, eta = 0.01,max_depth = 2, predict_type = "prob")
xgboost_lrn = po('imputemean') %>>%  po('scale')  %>>%  xgboost_lrn

learners = list(ranger_lrn, rpart_lrn, xgboost_lrn, featureless_lrn)

# Creating a Benchmark Grid and Benchmarking: 
bm_design_4 = benchmark_grid(tasks, learners, rsmp("cv", folds=3)) #Creating a benchmarking design
bench_4     = benchmark(bm_design_4, store_models = TRUE) 
print(bench_4) 
autoplot(bench_4, type = "boxplot") 

# Ranking the learners again
tab_4   = bench_4$aggregate(measures = msrs(c("classif.ce", "classif.bbrier", "classif.auc"))) 
ranks_4 = tab_4[, .(learner_id , rankce = rank(classif.ce),rankbrier = rank(classif.bbrier), rankauc=rank(-classif.auc), classif.ce, classif.bbrier, classif.auc), by = task_id]
print(ranks_4)


# Section 6: Visualization of Change in Evaluation Measures for all Benchmarks 

learner_names = c('ranger', 'rpart', 'xgboost', 'featureless')

# Defining R6 class for plotting the evaluation measures for all benchmarks

ErrorPlot = R6::R6Class("ErrorPlot",public = list(
  data     = NULL,
  error    = NULL,
  pipeline = NULL,
  learner  = NULL,
  y_lab    = NULL,
  gg_title = NULL,
  
  initialize = function(data, error, pipeline, learner, y_lab, gg_title) 
    {
      self$data = data
      self$error = error
      self$pipeline = pipeline
      self$learner = learner
      self$y_lab = y_lab
      self$gg_title = gg_title
    },
  
  plot = function() 
    {
      p = ggplot(self$data, aes(x = self$pipeline, y = self$error, color = self$learner)) +
      geom_line(aes(group = self$learner)) +
      geom_point(aes(group = self$learner)) +
      geom_text(aes(label = round(self$error, 2)),hjust = 1.5, vjust = -0.5) +
      theme_classic() +
      xlab("Preprocessing Pipeline") +
      ylab(self$y_lab) +
      ggtitle(self$gg_title) +
      scale_color_discrete(name = "Learner")
      print(p)
    }
))



# [PIMA Dataset] Visualizing change in Classification Error for all learners: Featureless Classifier, Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier 
# Pipeline breakdown: 
# Pipeline:             'mean'       -       Mean Imputation Pipeline and Scaling Pipeline
# Pipeline:             'hist'       -       Histogram Imputation and Scaling Pipeline
# Pipeline:             'varfil'     -       Mean Imputation and Variance Filter Feature Selection along with Scaling 
# Pipeline:             'notune'     -       Mean Imputation and Scaling Pipeline but with no hyperparameter optimization auto-tuning
 
pipeline_visualize_pima   = data.frame(pipeline = c('mean', 'mean','mean','mean', 'hist', 'hist', 'hist', 'hist', 'varfil','varfil','varfil','varfil','notune','notune','notune','notune'),
                                       learner  = c(learner_names, learner_names, learner_names, learner_names),
                                       error    = c(ranks_1$classif.ce[1:4], ranks_2$classif.ce[1:4], ranks_3$classif.ce[1:4], ranks_4$classif.ce[1:4]),
                                       brier    = c(ranks_1$classif.bbrier[1:4], ranks_2$classif.bbrier[1:4], ranks_3$classif.bbrier[1:4], ranks_4$classif.bbrier[1:4]),
                                       auc      = c(ranks_1$classif.auc[1:4], ranks_2$classif.auc[1:4], ranks_3$classif.auc[1:4], ranks_4$classif.auc[1:4]),
                                       rankce   = c(ranks_1$rankce[1:4], ranks_2$rankce[1:4], ranks_3$rankce[1:4], ranks_4$rankce[1:4]),
                                       rankbr   = c(ranks_1$rankbrier[1:4], ranks_2$rankbrier[1:4], ranks_3$rankbrier[1:4], ranks_4$rankbrier[1:4]),
                                       rankauc  = c(ranks_1$rankauc[1:4], ranks_2$rankauc[1:4], ranks_3$rankauc[1:4], ranks_4$rankauc[1:4])
                                       
)
pipeline_visualize_pima = pipeline_visualize_pima %>% group_by(pipeline, learner) %>% summarize(error = mean(error), brier= mean(brier), auc=mean(auc), rankce=mean(rankce), rankbr=mean(rankbr), rankauc=mean(rankauc))
plot_1ce      = ErrorPlot$new(data = pipeline_visualize_pima, error = pipeline_visualize_pima$error, pipeline = pipeline_visualize_pima$pipeline, learner = pipeline_visualize_pima$learner, y_lab = 'Classification Error', gg_title = 'CE: PIMA') 
plot_1br      = ErrorPlot$new(data = pipeline_visualize_pima, error = pipeline_visualize_pima$brier, pipeline = pipeline_visualize_pima$pipeline, learner = pipeline_visualize_pima$learner, y_lab = 'Brier Score', gg_title = 'Brier: PIMA') 
plot_1auc     = ErrorPlot$new(data = pipeline_visualize_pima, error = pipeline_visualize_pima$auc, pipeline = pipeline_visualize_pima$pipeline, learner = pipeline_visualize_pima$learner, y_lab = 'AUC Score', gg_title = 'AUC: PIMA') 
plot_1rankce  = ErrorPlot$new(data = pipeline_visualize_pima, error = pipeline_visualize_pima$rankce, pipeline = pipeline_visualize_pima$pipeline, learner = pipeline_visualize_pima$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(CE): PIMA') 
plot_1rankbr  = ErrorPlot$new(data = pipeline_visualize_pima, error = pipeline_visualize_pima$rankbr, pipeline = pipeline_visualize_pima$pipeline, learner = pipeline_visualize_pima$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(Brier): PIMA') 
plot_1rankauc = ErrorPlot$new(data = pipeline_visualize_pima, error = pipeline_visualize_pima$rankauc, pipeline = pipeline_visualize_pima$pipeline, learner = pipeline_visualize_pima$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(AUC): PIMA') 
p1ce          = plot_1ce$plot()
p1br          = plot_1br$plot()
p1auc         = plot_1auc$plot()
p1rankce      = plot_1rankce$plot()
p1rankbr      = plot_1rankbr$plot()
p1rankauc     = plot_1rankauc$plot()




# [Sonar Dataset] Visualizing change in Classification Error for all learners: Featureless Classifier, Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier
pipeline_visualize_sonar   = data.frame(pipeline = c('mean', 'mean','mean','mean', 'hist', 'hist', 'hist', 'hist', 'varfil','varfil','varfil','varfil','notune','notune','notune','notune'),
                                        learner  = c(learner_names, learner_names, learner_names, learner_names),
                                        error    = c(ranks_1$classif.ce[5:8], ranks_2$classif.ce[5:8], ranks_3$classif.ce[5:8], ranks_4$classif.ce[5:8]),
                                        brier    = c(ranks_1$classif.bbrier[5:8], ranks_2$classif.bbrier[5:8], ranks_3$classif.bbrier[5:8], ranks_4$classif.bbrier[5:8]),
                                        auc      = c(ranks_1$classif.auc[5:8], ranks_2$classif.auc[5:8], ranks_3$classif.auc[5:8], ranks_4$classif.auc[5:8]),
                                        rankce   = c(ranks_1$rankce[5:8], ranks_2$rankce[5:8], ranks_3$rankce[5:8], ranks_4$rankce[5:8]),
                                        rankbr   = c(ranks_1$rankbrier[5:8], ranks_2$rankbrier[5:8], ranks_3$rankbrier[5:8], ranks_4$rankbrier[5:8]),
                                        rankauc  = c(ranks_1$rankauc[5:8], ranks_2$rankauc[5:8], ranks_3$rankauc[5:8], ranks_4$rankauc[5:8])
                                        
)
pipeline_visualize_sonar = pipeline_visualize_sonar %>% group_by(pipeline, learner) %>% summarize(error = mean(error), brier= mean(brier), auc=mean(auc), rankce=mean(rankce), rankbr=mean(rankbr), rankauc=mean(rankauc))
plot_2ce      = ErrorPlot$new(data = pipeline_visualize_sonar, error = pipeline_visualize_sonar$error, pipeline = pipeline_visualize_sonar$pipeline, learner = pipeline_visualize_sonar$learner, y_lab = 'Classification Error', gg_title = 'CE: Sonar') 
plot_2br      = ErrorPlot$new(data = pipeline_visualize_sonar, error = pipeline_visualize_sonar$brier, pipeline = pipeline_visualize_sonar$pipeline, learner = pipeline_visualize_sonar$learner, y_lab = 'Brier Score', gg_title = 'Brier: Sonar') 
plot_2auc     = ErrorPlot$new(data = pipeline_visualize_sonar, error = pipeline_visualize_sonar$auc, pipeline = pipeline_visualize_sonar$pipeline, learner = pipeline_visualize_sonar$learner, y_lab = 'AUC Score', gg_title = 'AUC: Sonar') 
plot_2rankce  = ErrorPlot$new(data = pipeline_visualize_sonar, error = pipeline_visualize_sonar$rankce, pipeline = pipeline_visualize_sonar$pipeline, learner = pipeline_visualize_sonar$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(CE): Sonar') 
plot_2rankbr  = ErrorPlot$new(data = pipeline_visualize_sonar, error = pipeline_visualize_sonar$rankbr, pipeline = pipeline_visualize_sonar$pipeline, learner = pipeline_visualize_sonar$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(Brier): Sonar') 
plot_2rankauc = ErrorPlot$new(data = pipeline_visualize_sonar, error = pipeline_visualize_sonar$rankauc, pipeline = pipeline_visualize_sonar$pipeline, learner = pipeline_visualize_sonar$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(AUC): Sonar') 
p2ce          = plot_2ce$plot()
p2br          = plot_2br$plot()
p2auc         = plot_2auc$plot()
p2rankce      = plot_2rankce$plot()
p2rankbr      = plot_2rankbr$plot()
p2rankauc     = plot_2rankauc$plot()



# [BankNote Authentication Dataset] Visualizing change in Classification Error for all learners: Featureless Classifier, Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier
pipeline_visualize_bank   = data.frame(pipeline = c('mean', 'mean','mean','mean', 'hist', 'hist', 'hist', 'hist', 'varfil','varfil','varfil','varfil','notune','notune','notune','notune'),
                                       learner  = c(learner_names, learner_names, learner_names, learner_names),
                                       error    = c(ranks_1$classif.ce[9:12], ranks_2$classif.ce[9:12], ranks_3$classif.ce[9:12], ranks_4$classif.ce[9:12]),
                                       brier    = c(ranks_1$classif.bbrier[9:12], ranks_2$classif.bbrier[9:12], ranks_3$classif.bbrier[9:12], ranks_4$classif.bbrier[9:12]),
                                       auc      = c(ranks_1$classif.auc[9:12], ranks_2$classif.auc[9:12], ranks_3$classif.auc[9:12], ranks_4$classif.auc[9:12]),
                                       rankce   = c(ranks_1$rankce[9:12], ranks_2$rankce[9:12], ranks_3$rankce[9:12], ranks_4$rankce[9:12]),
                                       rankbr   = c(ranks_1$rankbrier[9:12], ranks_2$rankbrier[9:12], ranks_3$rankbrier[9:12], ranks_4$rankbrier[9:12]),
                                       rankauc  = c(ranks_1$rankauc[9:12], ranks_2$rankauc[9:12], ranks_3$rankauc[9:12], ranks_4$rankauc[9:12])
                                       
)
pipeline_visualize_bank = pipeline_visualize_bank %>% group_by(pipeline, learner) %>% summarize(error = mean(error), brier= mean(brier), auc=mean(auc), rankce=mean(rankce), rankbr=mean(rankbr), rankauc=mean(rankauc))
plot_3ce      = ErrorPlot$new(data = pipeline_visualize_bank, error = pipeline_visualize_bank$error, pipeline = pipeline_visualize_bank$pipeline, learner = pipeline_visualize_bank$learner, y_lab = 'Classification Error', gg_title = 'CE: BankNote Authentication') 
plot_3br      = ErrorPlot$new(data = pipeline_visualize_bank, error = pipeline_visualize_bank$brier, pipeline = pipeline_visualize_bank$pipeline, learner = pipeline_visualize_bank$learner, y_lab = 'Brier Score', gg_title = 'Brier: BankNote Authentication') 
plot_3auc     = ErrorPlot$new(data = pipeline_visualize_bank, error = pipeline_visualize_bank$auc, pipeline = pipeline_visualize_bank$pipeline, learner = pipeline_visualize_bank$learner, y_lab = 'AUC Score', gg_title = 'AUC: BankNote Authentication') 
plot_3rankce  = ErrorPlot$new(data = pipeline_visualize_bank, error = pipeline_visualize_bank$rankce, pipeline = pipeline_visualize_bank$pipeline, learner = pipeline_visualize_bank$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(CE): BankNote Authentication') 
plot_3rankbr  = ErrorPlot$new(data = pipeline_visualize_bank, error = pipeline_visualize_bank$rankbr, pipeline = pipeline_visualize_bank$pipeline, learner = pipeline_visualize_bank$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(Brier): BankNote Authentication') 
plot_3rankauc = ErrorPlot$new(data = pipeline_visualize_bank, error = pipeline_visualize_bank$rankauc, pipeline = pipeline_visualize_bank$pipeline, learner = pipeline_visualize_bank$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(AUC): BankNote Authentication') 
p3ce          = plot_3ce$plot()
p3br          = plot_3br$plot()
p3auc         = plot_3auc$plot()
p3rankce      = plot_3rankce$plot()
p3rankbr      = plot_3rankbr$plot()
p3rankauc     = plot_3rankauc$plot()



# [Blood Transfusion Service Center Dataset] Visualizing change in Classification Error for all learners: Featureless Classifier, Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier
pipeline_visualize_blood   = data.frame(pipeline = c('mean', 'mean','mean','mean', 'hist', 'hist', 'hist', 'hist', 'varfil','varfil','varfil','varfil','notune','notune','notune','notune'),
                                        learner  = c(learner_names, learner_names, learner_names, learner_names),
                                        error    = c(ranks_1$classif.ce[13:16], ranks_2$classif.ce[13:16], ranks_3$classif.ce[13:16], ranks_4$classif.ce[13:16]),
                                        brier    = c(ranks_1$classif.bbrier[13:16], ranks_2$classif.bbrier[13:16], ranks_3$classif.bbrier[13:16], ranks_4$classif.bbrier[13:16]),
                                        auc      = c(ranks_1$classif.auc[13:16], ranks_2$classif.auc[13:16], ranks_3$classif.auc[13:16], ranks_4$classif.auc[13:16]),
                                        rankce   = c(ranks_1$rankce[13:16], ranks_2$rankce[13:16], ranks_3$rankce[13:16], ranks_4$rankce[13:16]),
                                        rankbr   = c(ranks_1$rankbrier[13:16], ranks_2$rankbrier[13:16], ranks_3$rankbrier[13:16], ranks_4$rankbrier[13:16]),
                                        rankauc  = c(ranks_1$rankauc[13:16], ranks_2$rankauc[13:16], ranks_3$rankauc[13:16], ranks_4$rankauc[13:16])
                                        
                                        
)
pipeline_visualize_blood = pipeline_visualize_blood %>% group_by(pipeline, learner) %>% summarize(error = mean(error), brier= mean(brier), auc=mean(auc), rankce=mean(rankce), rankbr=mean(rankbr), rankauc=mean(rankauc))
plot_4ce      = ErrorPlot$new(data = pipeline_visualize_blood, error = pipeline_visualize_blood$error, pipeline = pipeline_visualize_blood$pipeline, learner = pipeline_visualize_blood$learner, y_lab = 'Classification Error', gg_title = 'CE: Blood Transfusion Service Center') 
plot_4br      = ErrorPlot$new(data = pipeline_visualize_blood, error = pipeline_visualize_blood$brier, pipeline = pipeline_visualize_blood$pipeline, learner = pipeline_visualize_blood$learner, y_lab = 'Brier Score', gg_title = 'Brier: Blood Transfusion Service Center') 
plot_4auc     = ErrorPlot$new(data = pipeline_visualize_blood, error = pipeline_visualize_blood$auc, pipeline = pipeline_visualize_blood$pipeline, learner = pipeline_visualize_blood$learner, y_lab = 'AUC Score', gg_title = 'AUC: Blood Transfusion Service Center') 
plot_4rankce  = ErrorPlot$new(data = pipeline_visualize_blood, error = pipeline_visualize_blood$rankce, pipeline = pipeline_visualize_blood$pipeline, learner = pipeline_visualize_blood$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(CE): Blood Transfusion Service Center') 
plot_4rankbr  = ErrorPlot$new(data = pipeline_visualize_blood, error = pipeline_visualize_blood$rankbr, pipeline = pipeline_visualize_blood$pipeline, learner = pipeline_visualize_blood$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(Brier): Blood Transfusion Service Center') 
plot_4rankauc = ErrorPlot$new(data = pipeline_visualize_blood, error = pipeline_visualize_blood$rankauc, pipeline = pipeline_visualize_blood$pipeline, learner = pipeline_visualize_blood$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(AUC): Blood Transfusion Service Center') 
p4ce          = plot_4ce$plot()
p4br          = plot_4br$plot()
p4auc         = plot_4auc$plot()
p4rankce      = plot_4rankce$plot()
p4rankbr      = plot_4rankbr$plot()
p4rankauc     = plot_4rankauc$plot()


# [Breast Cancer Wisconsin Dataset] Visualizing change in Classification Error for all learners: Featureless Classifier, Random Forest Classifier, Decision Tree Classifier, XGBoost Classifier
pipeline_visualize_breast   = data.frame(pipeline = c('mean', 'mean','mean','mean', 'hist', 'hist', 'hist', 'hist', 'varfil','varfil','varfil','varfil','notune','notune','notune','notune'),
                                         learner  = c(learner_names, learner_names, learner_names, learner_names),
                                         error    = c(ranks_1$classif.ce[17:20], ranks_2$classif.ce[17:20], ranks_3$classif.ce[17:20], ranks_4$classif.ce[17:20]),
                                         brier    = c(ranks_1$classif.bbrier[17:20], ranks_2$classif.bbrier[17:20], ranks_3$classif.bbrier[17:20], ranks_4$classif.bbrier[17:20]),
                                         auc      = c(ranks_1$classif.auc[17:20], ranks_2$classif.auc[17:20], ranks_3$classif.auc[17:20], ranks_4$classif.auc[17:20]),
                                         rankce   = c(ranks_1$rankce[17:20], ranks_2$rankce[17:20], ranks_3$rankce[17:20], ranks_4$rankce[17:20]),
                                         rankbr   = c(ranks_1$rankbrier[17:20], ranks_2$rankbrier[17:20], ranks_3$rankbrier[17:20], ranks_4$rankbrier[17:20]),
                                         rankauc  = c(ranks_1$rankauc[17:20], ranks_2$rankauc[17:20], ranks_3$rankauc[17:20], ranks_4$rankauc[17:20])
                                         
                                         
)
pipeline_visualize_breast = pipeline_visualize_breast %>% group_by(pipeline, learner) %>% summarize(error = mean(error), brier= mean(brier), auc=mean(auc), rankce=mean(rankce), rankbr=mean(rankbr), rankauc=mean(rankauc))
plot_5ce      = ErrorPlot$new(data = pipeline_visualize_breast, error = pipeline_visualize_breast$error, pipeline = pipeline_visualize_breast$pipeline, learner = pipeline_visualize_breast$learner, y_lab = 'Classification Error', gg_title = 'CE: Breast Cancer') 
plot_5br      = ErrorPlot$new(data = pipeline_visualize_breast, error = pipeline_visualize_breast$brier, pipeline = pipeline_visualize_breast$pipeline, learner = pipeline_visualize_breast$learner, y_lab = 'Brier Score', gg_title = 'Brier: Breast Cancer') 
plot_5auc     = ErrorPlot$new(data = pipeline_visualize_breast, error = pipeline_visualize_breast$auc, pipeline = pipeline_visualize_breast$pipeline, learner = pipeline_visualize_breast$learner, y_lab = 'AUC Score', gg_title = 'AUC: Breast Cancer') 
plot_5rankce  = ErrorPlot$new(data = pipeline_visualize_breast, error = pipeline_visualize_breast$rankce, pipeline = pipeline_visualize_breast$pipeline, learner = pipeline_visualize_breast$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(CE): Breast Cancer') 
plot_5rankbr  = ErrorPlot$new(data = pipeline_visualize_breast, error = pipeline_visualize_breast$rankbr, pipeline = pipeline_visualize_breast$pipeline, learner = pipeline_visualize_breast$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(Brier): Breast Cancer') 
plot_5rankauc = ErrorPlot$new(data = pipeline_visualize_breast, error = pipeline_visualize_breast$rankauc, pipeline = pipeline_visualize_breast$pipeline, learner = pipeline_visualize_breast$learner, y_lab = 'Rankings', gg_title = 'Change in Rankings(AUC): Breast Cancer') 
p5ce          = plot_5ce$plot()
p5br          = plot_5br$plot()
p5auc         = plot_5auc$plot()
p5rankce      = plot_5rankce$plot()
p5rankbr      = plot_5rankbr$plot()
p5rankauc     = plot_5rankauc$plot()



# Aggregating all the plots into one Matrix Plot for easier comparison
grid.arrange(p1ce,  p2ce,  p3ce,  p4ce ,  p5ce ,  ncol = 3) # Matrix plot for Classification Error on all datasets, pre-processing and learner pipelines
grid.arrange(p1br,  p2br,  p3br,  p4br ,  p5br ,  ncol = 3) # Matrix plot for Brier Score on all datasets, pre-processing and learner pipelines
grid.arrange(p1auc, p2auc, p3auc, p4auc , p5auc , ncol = 3) # Matrix plot for Brier Score on all datasets, pre-processing and learner pipelines

grid.arrange(p1rankce,  p2rankce,  p3rankce,  p4rankce ,  p5rankce ,  ncol = 3) # Matrix plot for Rankings of Learners wrt Classification Error on all datasets, pre-processing and learner pipelines
grid.arrange(p1rankbr,  p2rankbr,  p3rankbr,  p4rankbr ,  p5rankbr ,  ncol = 3) # Matrix plot for Rankings of Learners wrt Brier Score on all datasets, pre-processing and learner pipelines
grid.arrange(p1rankauc, p2rankauc, p3rankauc, p4rankauc , p5rankauc , ncol = 3) # Matrix plot for Rankings of Learners wrt AUC Score on all datasets, pre-processing and learner pipelines



# These are the Classification Error , Brier Score, AUC score for different datasets, learners and pre-processing pipelines
# One can compare the results obtained in these tables to check for reproducibility
pipeline_visualize_pima
pipeline_visualize_sonar
pipeline_visualize_bank
pipeline_visualize_blood
pipeline_visualize_breast





