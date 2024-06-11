import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from time import ctime,perf_counter

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score, roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot

data_dir = "./data/"



  ################
###    Utility   ###
  ################

def estimatedTime(model, params, X_train, y_train):
    
    s = perf_counter()
    for i in range(5):
        model.fit(X_train, y_train)
    s = (perf_counter() - s) / 5
    
    num = 1
    for k,v in params.items():
        num *= len(v)
        
    print("Estimated time : {:.3f} minutes to run ".format((s*num)/60), num, " models")
    
def getBestModel(models, X_test, y_test, scoring):
    scores = []
    for m in models:
        y_pred = m.predict(X_test)
        
        if scoring == "accuracy":
            scores.append([accuracy_score(y_test,y_pred)])
        
        elif scoring == "f1":
            scores.append([f1_score(y_test,y_pred)])
            
    return models[np.argmax(scores)]



  #################
###    PLOTTING   ###
  #################
    
def plotLogisticCoefficients(model, columns, fig=None, subplot_index=111): # X_train.coluns
    
    if not fig:
        fig = plt.figure()
    
    values = []
    if str(type(model)).split("'")[1].split(".")[0] == "sklearn":
        values = model.coef_[0]
    else:
        values = model.params.values
    
    ax = fig.add_subplot(subplot_index)
    coefs = pd.Series(model.coef_[0], index=columns)
    coefs.sort_values(inplace=True)
    plt.title("Coefficients")
    coefs.plot(kind="bar")

def plotFeatureImportances(model, columns, fig, subplot_index):
    
    ncol = len(columns)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    ax = fig.add_subplot(subplot_index)
    plt.title("Feature Importances")
    plt.bar(range(ncol),
            importances[indices],
            align="center",
            alpha=0.5)

    plt.xticks(range(ncol),
               columns.values[indices], rotation=90)
    
def plotClassification(model, X_train, X_test, y_train, y_test, fig, subplot_indexes):
    
    model_type = str(type(model)).split("'")[1].split(".")

    if model_type[0] == "sklearn":
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
    else:
        y_pred_test = np.where(model.predict(X_test) < 0.5, 0, 1)
        y_pred_train = np.where(model.predict(X_train) < 0.5, 0, 1)
    
    cm = confusion_matrix(y_test, y_pred_test)
    fpr, tpr, thresh = roc_curve(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_pred_test)

    print("Train Accuracy: {:6f}".format(accuracy_score(y_train, y_pred_train)))
    print("Test Accuracy:  {:6f}".format(accuracy_score(y_test, y_pred_test)))
    print("F1 score:       {:6f}".format(f1_score(y_test, y_pred_test)))
    print("AUC:            {:6f}".format(auc))

    # confusion matrix
    ax1 = fig.add_subplot(subplot_indexes[0])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax1)
    ax1.set_ylabel("Real value")
    ax1.set_xlabel("Predicted value")
    ax1.set_title("Confusion Matrix")

    # roc plot
    ax2 = fig.add_subplot(subplot_indexes[1])
    ax2.plot(fpr, tpr, label="AUC = %0.2f" % auc)
    ax2.plot([0, 1], [0, 1], "r--")
    ax2.set_ylabel("False Positive Rate")
    ax2.set_xlabel("True Positive Rate")
    ax2.set_title("AUC")
    ax2.legend(loc="lower right")

def modelEvalClass(model, X_train, X_test, y_train, y_test):
    
    model_type = str(type(model)).split("'")[1].split(".")
    
    if model_type[3] == "LogisticRegression":
        fig = plt.figure(figsize=(12,10), layout="tight")
        plotClassification(model, X_train, X_test, y_train, y_test, fig, [221,222])
        plotLogisticCoefficients(model, X_train.columns, fig, 212)
    elif model_type[1] == "ensemble":
        fig = plt.figure(figsize=(12,10), layout="tight")
        plotClassification(model, X_train, X_test, y_train, y_test, fig, [221,222])
        plotFeatureImportances(model, X_train.columns, fig, 212)
    else:
        fig = plt.figure(layout="tight")
        plotClassification(model, X_train, X_test, y_train, y_test, fig, [121,122])
        
def modelEvalReg(model, X_train, X_test, y_train, y_test):
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    print("Training set:")
    print("\tR2 score            :",r2_score(y_train,y_pred_train))
    print("\tRoot Mean error     :",np.sqrt(mean_squared_error(y_train,y_pred_train)))
    print("\tMean Absolute error :",mean_absolute_error(y_train,y_pred_train))

    print("Testing set:")
    print("\tR2 score            :",r2_score(y_test,y_pred_test))
    print("\tRoot Mean error     :",np.sqrt(mean_squared_error(y_test,y_pred_test)))
    print("\tMean Absolute error :",mean_absolute_error(y_test,y_pred_test))

        
        

  ##############################
###    CLASSIFICATION MODELS   ###
  ##############################

def kNNClass(X_train, X_test, y_train, y_test,
       
        # grid search params
        scoring="accuracy",
        cv=5,
        verbose_search=0,
        n_jobs_search=-1,
        
       # kNN params
        n_neighbors=[],
        weights=["uniform","distance"],
        algorithm=["ball_tree", "kd_tree", "brute"],
        leaf_size=[30],
        p=[1,2,3,4],
        metric=["minkowski"],
        metric_params=[None],
        n_jobs_model=[None]
):
    
    # set n_neighbors
    if not n_neighbors:
        # 2 -> 2*sqrt(m)
        n_neighbors = np.arange(2, int(np.sqrt(X_train.shape[0]))*2)
    
    params = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
        "metric": metric,
        "metric_params": metric_params,
        "n_jobs": n_jobs_model
    }
    
    estimatedTime(KNeighborsClassifier(n_neighbors=max(n_neighbors)), params, X_train, y_train)

    grid = GridSearchCV(estimator= KNeighborsClassifier(), #<- model object
                        param_grid= params,                #<- param grid
                        scoring= scoring,                  #<- scoring param
                        cv= cv,                            #<- 5-fold CV
                        verbose= verbose_search,           #<- silence lengthy output to console
                        n_jobs= n_jobs_search              #<- use all available processors
    )
    
    s = perf_counter()
    grid.fit(X_train, y_train)
    print("Actual time    : {:.3f} minutes\n".format((perf_counter() - s)/60))

    print("Best params    : ", grid.best_params_,"\n")
    
    model = KNeighborsClassifier(**grid.best_params_).fit(X_train, y_train)
    modelEvalClass(model, X_train, X_test, y_train, y_test)
    
    return model
    
    
def LogisticClass(X_train, X_test, y_train, y_test,
       
        # grid search params
        scoring="accuracy",
        cv=5,
        verbose_search=0,
        n_jobs_search=-1,
        
       # logistic params
        penalty=["l1", "l2", "elasticnet", None],
        dual=[False],
        tol=[1e-4],
        C=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        fit_intercept=[True],
        intercept_scaling=[1],
        class_weight=["balanced", None],
        random_state=[42],
        solver=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        max_iter=[7000],
        multi_class=["ovr"], #"multinomial"
        verbose_model=[0],
        warm_start=[True,False],
        l1_ratio=np.arange(0,1.1,0.1)
):
    
    model_params = [{
        "penalty": penalty,
        "dual": dual,
        "tol": tol,
        "C": C,
        "fit_intercept": fit_intercept,
        "intercept_scaling": intercept_scaling,
        "class_weight": class_weight,
        "random_state": random_state,
        "solver": solver,
        "max_iter": max_iter,
        "multi_class": multi_class, #"multinomial"
        "verbose": verbose_model,
        "warm_start": warm_start,
    } for i in range(len(penalty))]

    
    for i,pen in enumerate(penalty):
        model_params[i]["penalty"] = [pen]
        if pen == "l1":
            if multi_class == "ovr":
                model_params[i]["solver"] = ["liblinear", "saga"]
            else:
                model_params[i]["solver"] = ["saga"]
        elif pen == "l2":
            if multi_class != "ovr":
                model_params[i]["solver"] = ["lbfgs", "newton-cg", "sag", "saga"]
        elif pen == "elasticnet":
            model_params[i]["solver"] = ["saga"]
            model_params[i]["l1_ratio"] = l1_ratio
        else:
            model_params[i]["solver"] = ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]
            model_params[i].pop("C")
                  
    search_results = []
    s1 = perf_counter()
    for i,params in enumerate(model_params):
        grid = GridSearchCV(estimator= LogisticRegression(), #<- model object
                      param_grid= params,                     #<- param grid
                      scoring= scoring,                        #<- scoring param
                      cv= cv,                                 #<- 5-fold CV
                      verbose= verbose_search,               #<- silence lengthy output to console
                      n_jobs= n_jobs_search)                  #<- use all available processors

        s2 = perf_counter()
        grid.fit(X_train, y_train)
        print("penalty",penalty[i]," : {:.3f} minutes".format((perf_counter()-s2)/60))
        
        search_results.append(
            LogisticRegression(**grid.best_params_).fit(X_train,y_train)
        )
    print("Full time    : {:.3f} minutes\n".format((perf_counter() - s1)/60))

    
    print("Best params    : ", grid.best_params_,"\n")
    
    model = getBestModel(search_results, X_test, y_test, scoring)
    modelEvalClass(model, X_train, X_test, y_train, y_test)

    return model

"""
Perform a forward-backward feature selection 
based on p-value from statsmodels.api.OLS
Arguments:
    X - pandas.DataFrame with candidate features
    y - list-like with the target
    initial_list - list of features to start with (column names of X)
    threshold_in - include a feature if its p-value < threshold_in
    threshold_out - exclude a feature if its p-value > threshold_out
    verbose - whether to print the sequence of inclusions and exclusions
Returns: list of selected features 
Always set threshold_in < threshold_out to avoid infinite looping.
See https://en.wikipedia.org/wiki/Stepwise_regression for the details
"""
def StepwiseLogisticClass(X_train, X_test, y_train, y_test,

    # stepwise arguments
    initial_list=[],
    threshold_in=0.01,
    threshold_out = 0.05,
    verbose=True
):
    
    included = list(initial_list)
    s = perf_counter()
    while True:
        changed=False
        # forward step
        excluded = list(set(X_train.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype='float64')
        for new_column in excluded:
            model = sm.Logit(y_train, X_train[included+[new_column]]).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.Logit(y_train, X_train[included]).fit(disp=0)
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    
    print("Best features  : ", included,"\n")

    model = LogisticClass(X_train[included], X_test[included], y_train, y_test, scoring="f1")

    return model


def RandomForestClass(X_train, X_test, y_train, y_test,
       
        # grid search params
        scoring="accuracy",
        cv=5,
        verbose_search=0,
        n_jobs_search=-1,
        
       # random forest params
        n_estimators=[50],
        criterion=["gini", "entropy"],
        max_depth=np.append(np.arange(1,10), None),
        min_samples_split=[0.1,2],
        min_samples_leaf=[0.1,1],
        min_weight_fraction_leaf=[0],
        max_features=["sqrt", "log2", None],
        max_leaf_nodes=[None],
        min_impurity_decrease=[0],
        bootstrap=[True, False],
        oob_score=[True, False],
        random_state=[42],
        verbose_model=[0],
        warm_start=[True,False],
        class_weight=["balanced", "balanced_subsample", None],
        ccp_alpha=[0],
        max_samples=np.arange(0.1,1,0.1)
):
    
    model_params = [{
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_features": max_features,
        "max_leaf_nodes": max_leaf_nodes,
        "min_impurity_decrease": min_impurity_decrease,
        "bootstrap": bootstrap,
        "oob_score": oob_score,
        "random_state": random_state,
        "verbose": verbose_model,
        "class_weight": class_weight,
        "ccp_alpha": ccp_alpha,
        "max_samples": max_samples
    } for i in range(len(warm_start))]
          
    for i,w in enumerate(warm_start):
        if w:
            model_params[i]["warm_start"] = [True]
            model_params[i]["class_weight"] = [None]
        else:
            model_params[i]["warm_start"] = [False]
            
    search_results = []
    s1 = perf_counter()
    for i,params in enumerate(model_params):
        grid = GridSearchCV(estimator= RandomForestClassifier(), #<- model object
                            param_grid= params,                  #<- param grid
                            scoring= scoring,                    #<- scoring param
                            cv= cv,                              #<- 5-fold CV
                            verbose= verbose_search,             #<- silence lengthy output to console
                            n_jobs= n_jobs_search)               #<- use all available processors

        s2 = perf_counter()
        grid.fit(X_train, y_train)
        print("warm_start:",warm_start[i],",  {:.3f} minutes".format((perf_counter()-s2)/60))
        
        search_results.append(
            RandomForestClassifier(**grid.best_params_).fit(X_train,y_train)
        )
    print("Full time    : {:.3f} minutes\n".format((perf_counter() - s1)/60))

        
    print("Best params    : ", grid.best_params_,"\n")

    model = getBestModel(search_results, X_test, y_test, scoring)
    modelEvalClass(model, X_train, X_test, y_train, y_test)

    return model
    
    
    
def GradientBoostClass(X_train, X_test, y_train, y_test,
       
        # grid search params
        scoring="accuracy",
        cv=5,
        verbose_search=0,
        n_jobs_search=-1,
        
       # random forest params
        loss=["log_loss", "exponential"],
        learning_rate=[0.01],
        n_estimators=np.arange(50,250,50),
        subsample=[1],
        criterion=["friedman_mse", "squared_error"],
        min_samples_split=[0.1,0.2],
        min_samples_leaf=[1],
        min_weight_fraction_leaf=[0],
        max_depth=np.arange(1,8),
        min_impurity_decrease=[0],
        init=[None],
        random_state=[42],
        max_features=["sqrt", "log2", None],
        verbose_model=[0],
        max_leaf_nodes=[None],
        warm_start=[True,False],
        validation_fraction=[0.1],
        n_iter_no_change=[None],
        tol=[1e-4],
        ccp_alpha=[0]
):
    model_params = {
        "loss": loss,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_depth": max_depth,
        "min_impurity_decrease": min_impurity_decrease,
        "init": init,
        "random_state": random_state,
        "max_features": max_features,
        "verbose": verbose_model,
        "max_leaf_nodes": max_leaf_nodes,
        "warm_start": warm_start,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
        "tol": tol,
        "ccp_alpha": ccp_alpha
    }

    grid = GridSearchCV(estimator= GradientBoostingClassifier(), #<- model object
                        param_grid= model_params,                #<- param grid
                        scoring= scoring,                        #<- scoring param
                        cv= cv,                                  #<- 5-fold CV
                        verbose= verbose_search,                 #<- silence lengthy output to console
                        n_jobs= n_jobs_search)                   #<- use all available processors

    s = perf_counter()
    grid.fit(X_train, y_train)
    print("Full time    : {:.3f} minutes\n".format((perf_counter() - s)/60))
    
    print("Best params    : ", grid.best_params_,"\n")

    model = GradientBoostingClassifier(**grid.best_params_).fit(X_train, y_train)
    modelEvalClass(model, X_train, X_test, y_train, y_test)

    return model



def SupportVectorClass(X_train, X_test, y_train, y_test,
       
        # grid search params
        scoring="accuracy",
        cv=5,
        verbose_search=0,
        n_jobs_search=-1,
        
       # support vector params
        C=[1],
        kernel=["linear", "poly", "rbf", "sigmoid"],
        degree=np.arange(2,6),
        gamma=["auto"], #"scale",
        coef0=[0],
        shrinking=[True],
        probability=[False],
        tol=[1e-3],
        cache_size=[200],
        class_weight=[None],
        verbose_model=[False],
        max_iter=[1], #-1
        decision_function_shape=["ovr"],
        break_ties=[False],
        random_state=[42]
):
    model_params = {
        "C": C,
        "kernel": kernel,
        "degree": degree,
        "gamma": gamma,
        "coef0": coef0,
        "shrinking": shrinking,
        "probability": probability,
        "tol": tol,
        "cache_size": cache_size,
        "class_weight": class_weight,
        "verbose": verbose_model,
        "max_iter": max_iter,
        "decision_function_shape": decision_function_shape,
        "break_ties": break_ties,
        "random_state": random_state
    }
    
    s1 = perf_counter()
    for k in kernel:
        model_params["kernel"] = [k]
        
        grid = GridSearchCV(estimator= SVC(),                        #<- model object
                            param_grid= model_params,                #<- param grid
                            scoring= scoring,                        #<- scoring param
                            cv= cv,                                  #<- 5-fold CV
                            verbose= verbose_search,                 #<- silence lengthy output to console
                            n_jobs= n_jobs_search)                   #<- use all available processors

        s = perf_counter()
        grid.fit(X_train, y_train)
        print(k,(perf_counter()-s)/60," minutes")
    print("Full time    : {:.3f} minutes\n".format((perf_counter() - s1)/60))

    
    print("Best params    : ", grid.best_params_,"\n")
    
    model = SVC(**grid.best_params_).fit(X_train, y_train)
    modelEvalClass(model, X_train, X_test, y_train, y_test)
    
    return model



  ##########################
###    REGRESSION MODELS   ###
  ##########################

    
  #######################
###    CLUSTER MODELS   ###
  #######################    