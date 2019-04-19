# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:15:32 2019

@author: Owner
"""
# importing modules used
import time
import winsound
import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import linear_model
import pydot
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# start timer to show how long program takes to run
start_time = time.time()

# set a random seed as to allow for the same trees obtained to be used on "lock box" set in future
# also allows for repeatable results
random.seed(0)


# timer to indicate time remaining
def timer(timestamp_1, timestamp_2, t, iterations):
    rf_time_diff = timestamp_2 - timestamp_1
    rf_time_rem = rf_time_diff * (iterations - t)
    rf_min_rem = int(rf_time_rem // 60)
    rf_sec_rem = int(rf_time_rem - 60 * rf_min_rem)
    if rf_min_rem == 1:
        minute = "min"
    else:
        minute = "mins"
    if rf_sec_rem == 1:
        second = "sec"
    else:
        second = "secs"
    print("time remaining:", rf_min_rem, minute, rf_sec_rem, second)


# Split the data into training and test sets
def data_split(random_state, test_size=0.20):

    # import data set (must be in order of picks)
    features = pd.read_csv('Data_WS_1_TNT.csv')

    # drop name and pick from data set
    features = features.drop('Name', axis=1)
    features = features.drop('Pick', axis=1)

    # Create indicator variables
    features = pd.get_dummies(features)

    # gather the labels of the rows
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    # identify the objective variable
    objective_index = feature_list.index('Peak WSPG')

    # separate the labels from the training and test sets
    labels = np.array(features[:, objective_index])

    # remove the objective variable from the features and feature list
    features = np.delete(features, objective_index, 1)
    feature_list.remove("Peak WSPG")

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size,
                                                                                random_state=random_state)

    # return the split data (features and labels) as well as the list of variables
    return [train_features, test_features, train_labels, test_labels, feature_list]


# run the random forest model and report the predictions (can also optionally report the importances)
def rand_forest(train_features, test_features, train_labels, feature_list, runs=1, n_estimators=1000,
                max_depth=None, min_samples_split=2, max_features="auto", get_importances=False, iteration=0):

    sorted_importances = []

    # run the random forest model a certain number of times (runs) for more accuracy
    for b in range(runs):

        # to track progress of the program
        # print(b)

        # Instantiate model with chosen parameters
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                   max_features=max_features, random_state=b)

        # Train the model on training data
        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)

        # build matrix to keep track of predictions
        if b == 0:
            predictions_matrix = np.array(predictions)
        else:
            predictions_matrix = np.column_stack((predictions_matrix, predictions))

        if get_importances:
            # Get numerical feature importances
            importances = list(rf.feature_importances_)

            # Label and sort the importances
            sorted_importances = np.column_stack((feature_list, importances))
            sorted_importances = np.flip(sorted_importances[sorted_importances[:, 1].argsort()], 0)
            print(sorted_importances)

        # toggle to turn off tree figure creation
        export_trees = False

        # create decision tree pictures, but only for first iteration, first run for the given parameters
        if b == 0 and iteration == 0 and export_trees is True:
            # Pull out three trees from the forest
            tree1 = rf.estimators_[0]
            tree2 = rf.estimators_[500]
            tree3 = rf.estimators_[999]

            # Export the image to a dot file
            export_graphviz(tree1, out_file='tree1.dot', feature_names=feature_list, rounded=True, precision=1)
            export_graphviz(tree2, out_file='tree2.dot', feature_names=feature_list, rounded=True, precision=1)
            export_graphviz(tree3, out_file='tree3.dot', feature_names=feature_list, rounded=True, precision=1)

            # Use dot file to create a graph
            (graph1,) = pydot.graph_from_dot_file('tree1.dot')
            (graph2,) = pydot.graph_from_dot_file('tree2.dot')
            (graph3,) = pydot.graph_from_dot_file('tree3.dot')

            # Create png file names to indicate parameters used
            pic1 = 'md-' + str(max_depth) + '_mss-' + str(min_samples_split) + '_mf-' + str(max_features) + '_tree1.png'
            pic2 = 'md-' + str(max_depth) + '_mss-' + str(min_samples_split) + '_mf-' + str(max_features) + '_tree2.png'
            pic3 = 'md-' + str(max_depth) + '_mss-' + str(min_samples_split) + '_mf-' + str(max_features) + '_tree3.png'

            # Write graph to a png file
            graph1.write_png(pic1)
            graph2.write_png(pic2)
            graph3.write_png(pic3)

    return [predictions_matrix, sorted_importances]


def lasso(train_features, test_features, train_labels, test_labels, runs=1, alpha=1.0, tol=0.0001):
    for b in range(runs):

        # Instantiate model with chosen parameters
        clf = linear_model.Lasso(alpha=alpha, max_iter=100000, tol=tol, random_state=b)

        # Train the model on training data
        clf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = clf.predict(test_features)

        # calculate R^2
        lasso_score = clf.score(test_features, test_labels)

        # build matrix to keep track of predictions
        if b == 0:
            predictions_matrix = np.array(predictions)
            lasso_score_matrix = np.array(lasso_score)
        else:
            predictions_matrix = np.column_stack((predictions_matrix, predictions))
            lasso_score_matrix = np.column_stack((lasso_score_matrix, lasso_score))

    lasso_score_avg = np.average(lasso_score_matrix)

    # print(clf.coef_)
    # print(clf.sparse_coef_)

    return [predictions_matrix, lasso_score_avg]


def lin_reg(train_features, test_features, train_labels, test_labels):
    # create linear regression object
    regr = linear_model.LinearRegression()

    # fit the model on the training set
    regr.fit(train_features, train_labels)

    # make predictions
    predictions = regr.predict(test_features)

    # the coefficients
    # print('Coefficients: \n', regr.coef_)

    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(test_labels, predictions))

    lin_reg_r2_score = r2_score(test_labels, predictions)

    return [predictions, lin_reg_r2_score]


# first version of the evaluation procedure. Returns performance of average objective outcome for the first nth picks
def eval_procedure_rmse(test_labels, predictions_matrix):

    # calculate errors and root mean square error
    if len(np.shape(predictions_matrix)) == 1:
        errors = (predictions_matrix - test_labels) ** 2
    else:
        errors = (np.average(predictions_matrix, axis=1) - test_labels) ** 2
    rmse = (np.average(errors)) ** 0.5
    root_mean_square_error = round(rmse, 4)

    return root_mean_square_error


def rand_forest_eval_1_main(iterations=1, runs=5, max_depth=None, min_samples_split=2, max_features="auto"):

    for t in range(iterations):
        # to track progress of the program
        print('-', t)

        # timer to indicate estimated time remaining
        if t > 0:
            timestamp_2 = time.time()
            timer(timestamp_1, timestamp_2, t, iterations)
        timestamp_1 = time.time()

        # obtain split data
        output_data_split = data_split(t)

        train_features = output_data_split[0]
        test_features = output_data_split[1]
        train_labels = output_data_split[2]
        test_labels = output_data_split[3]
        feature_list = output_data_split[4]

        output_rand_forest = rand_forest(train_features, test_features, train_labels, feature_list, runs,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         max_features=max_features, iteration=t)
        predictions_matrix = output_rand_forest[0]
        root_mean_square_error = eval_procedure_rmse(test_labels, predictions_matrix)
        print(root_mean_square_error)

        if t == 0:
            total_root_mean_square_error = np.array(root_mean_square_error)
        else:
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)

    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)

    return avg_total_root_mean_square_error


def lasso_eval_1_main(iterations=1, runs=5, alpha=1.0, tol=0.0001):

    for t in range(iterations):

        # obtain split data
        output_data_split = data_split(t)

        train_features = output_data_split[0]
        test_features = output_data_split[1]
        train_labels = output_data_split[2]
        test_labels = output_data_split[3]
        feature_list = output_data_split[4]

        # ========
        output_lasso = lasso(train_features, test_features, train_labels, test_labels, runs=runs, alpha=alpha, tol=tol)
        predictions_matrix = output_lasso[0]
        lasso_score_avg = output_lasso[1]
        # =====

        root_mean_square_error = eval_procedure_rmse(test_labels, predictions_matrix)

        if t == 0:
            total_root_mean_square_error = np.array(root_mean_square_error)
            total_lasso_score_avg = np.array(lasso_score_avg)
        else:
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)
            total_lasso_score_avg = np.append(total_lasso_score_avg, lasso_score_avg)

    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
        avg_total_lasso_score_avg = total_lasso_score_avg
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
        avg_total_lasso_score_avg = round(np.average(total_lasso_score_avg), 4)
    print(avg_total_root_mean_square_error)
    print(avg_total_lasso_score_avg)
    print('MAX', max(total_lasso_score_avg))
    print('MIN', min(total_lasso_score_avg))

    return [avg_total_root_mean_square_error, avg_total_lasso_score_avg]


def lin_reg_eval_1_main(iterations=1):

    for t in range(iterations):

        # obtain split data
        output_data_split = data_split(t)

        train_features = output_data_split[0]
        test_features = output_data_split[1]
        train_labels = output_data_split[2]
        test_labels = output_data_split[3]
        feature_list = output_data_split[4]

        # ========
        output_lin_reg = lin_reg(train_features, test_features, train_labels, test_labels)
        predictions = output_lin_reg[0]
        lin_reg_r2_score = output_lin_reg[1]
        # ========

        root_mean_square_error = eval_procedure_rmse(test_labels, predictions)

        if t == 0:
            total_root_mean_square_error = np.array(root_mean_square_error)
            total_lin_reg_r2_score = np.array(lin_reg_r2_score)
        else:
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)
            total_lin_reg_r2_score = np.append(total_lin_reg_r2_score, lin_reg_r2_score)
    print('MAX', max(total_lin_reg_r2_score))
    print('MIN', min(total_lin_reg_r2_score))

    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
        avg_total_lin_reg_r2_score = total_lin_reg_r2_score
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
        avg_total_lin_reg_r2_score = round(np.average(total_lin_reg_r2_score), 4)

    print(avg_total_root_mean_square_error)
    print(avg_total_lin_reg_r2_score)

    return [avg_total_root_mean_square_error, avg_total_lin_reg_r2_score]


def ensemble_eval_1_main(iterations=1, runs_rf=5, max_depth=None, min_samples_split=2, max_features="auto",
                         runs_las=5, alpha=1.0, tol=0.0001, weight_rf=1, weight_las=1, weight_lr=1):

    for t in range(iterations):
        print(t, '/', iterations)

        # timer to indicate estimated time remaining
        if t > 0:
            timestamp_2 = time.time()
            timer(timestamp_1, timestamp_2, t, iterations)
        timestamp_1 = time.time()

        # obtain split data
        output_data_split = data_split(t)

        train_features = output_data_split[0]
        test_features = output_data_split[1]
        train_labels = output_data_split[2]
        test_labels = output_data_split[3]
        feature_list = output_data_split[4]

        # obtain random forest output
        output_rand_forest = rand_forest(train_features, test_features, train_labels, feature_list, runs=runs_rf,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         max_features=max_features, iteration=t)
        # obtain random forest predictions matrix
        predictions_matrix_rf = output_rand_forest[0]
        # collapse the prediction matrix into the averages
        if len(np.shape(predictions_matrix_rf)) == 1:
            predictions_rf = predictions_matrix_rf
        else:
            predictions_rf = np.average(predictions_matrix_rf, axis=1)

        # obtain lasso output
        output_lasso = lasso(train_features, test_features, train_labels, test_labels, runs=runs_las, alpha=alpha,
                             tol=tol)
        # obtain lasso predictions matrix
        predictions_matrix_las = output_lasso[0]
        # collapse the prediction matrix into the averages
        if len(np.shape(predictions_matrix_las)) == 1:
            predictions_las = predictions_matrix_las
        else:
            predictions_las = np.average(predictions_matrix_las, axis=1)

        # obtain linear regression output
        output_lin_reg = lin_reg(train_features, test_features, train_labels, test_labels)
        # obtain linear regression predictions
        predictions_lr = output_lin_reg[0]

        # create ensemble predictions matrix
        predictions_matrix_en = np.column_stack((predictions_rf, predictions_las, predictions_lr))

        # create weighted average ensemble prediction
        total_weight = weight_rf + weight_las + weight_lr
        predictions_en = np.zeros((len(predictions_matrix_en), 1))

        for e in range(len(predictions_en)):
            predictions_en[e] = (weight_rf / total_weight) * predictions_matrix_en[e, 0] + \
                                (weight_las / total_weight) * predictions_matrix_en[e, 1] + \
                                (weight_lr / total_weight) * predictions_matrix_en[e, 2]

        # evaluate predictions
        ensemble_r2_score = r2_score(test_labels, predictions_en)
        root_mean_square_error = eval_procedure_rmse(test_labels, predictions_en)

        # create final evaluation and average total root mean square error outputs
        if t == 0:
            total_root_mean_square_error = np.array(root_mean_square_error)
            ensemble_r2_score_array = np.array(ensemble_r2_score)
        else:
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)
            ensemble_r2_score_array = np.append(ensemble_r2_score_array, ensemble_r2_score)
    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
        average_ensemble_r2_score = ensemble_r2_score_array
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
        average_ensemble_r2_score = round(np.average(ensemble_r2_score_array), 4)

    print(avg_total_root_mean_square_error)
    print('R2:', average_ensemble_r2_score)

    return [avg_total_root_mean_square_error, average_ensemble_r2_score]


# MODEL SELECT (set all to 1 for ensemble):
rf = 1
las = 1
lr = 1

# run random forest model
if rf == 1 and las == 0 and lr == 0:
    # parameters used (random forest)
    iterations = 5
    runs = 5
    max_depth = 6  # options: None or int
    min_samples_split = 25  # options: int (min of 2, which is the default) or float(fraction)
    max_features = "auto"  # options: auto(default, =n_features), sqrt, log2, int, float(fraction)

    # running the output
    output_rmse = rand_forest_eval_1_main(iterations, runs, max_depth, min_samples_split, max_features)
    print()
    print("Random Forest Results:")
    print("RMSE:", output_rmse)


# run lasso regression model
if las == 1 and rf == 0 and lr == 0:
    # parameters used (lasso)
    iterations = 100
    runs = 5
    alpha = 0.13  # options: float. WARNING: do not set this to 0
    tol = 0.000001

    # running the output
    output = lasso_eval_1_main(iterations, runs, alpha, tol)
    output_rmse = output[0]
    output_r2 = output[1]
    print()
    print("Lasso Regression Results:")
    print("RMSE:", output_rmse)
    print("R2:", output_r2)


# run linear regression
if lr == 1 and las == 0 and rf == 0:
    # parameters used (linear regression)
    iterations = 100

    # running the output
    output = lin_reg_eval_1_main(iterations)
    output_rmse = output[0]
    output_r2 = output[1]
    print()
    print("Linear Regression Results:")
    print("RMSE:", output_rmse)
    print("R2:", output_r2)


# run ensemble
if lr == 1 and las == 1 and rf == 1:
    # parameters used (general)
    iterations = 100
    weight_rf = 2  # ensemble prediction weight on random forest prediction
    weight_las = 3  # ensemble prediction weight on lasso prediction
    weight_lr = 1  # ensemble prediction weight on linear regression prediction

    # parameters used (random forest)
    runs_rf = 5
    max_depth = 6  # options: None or int
    min_samples_split = 25  # options: int (min of 2, which is the default) or float(fraction)
    max_features = "auto"  # options: auto(default, =n_features), sqrt, log2, int, float(fraction)

    # parameters used (lasso)
    runs_las = 5
    alpha = 0.13  # options: float. WARNING: do not set this to 0
    tol = 0.000001

    # running the output
    output = ensemble_eval_1_main(iterations, runs_rf, max_depth, min_samples_split, max_features,
                                  runs_las, alpha, tol, weight_rf, weight_las, weight_lr)
    output_rmse = output[0]
    output_r2 = output[1]
    print()
    print("Ensemble Results:")
    print("RMSE:", output_rmse)
    print("R2:", output_r2)


# print result of how long program takes to run
print()
print("My program took", time.time() - start_time, "to run")

# sound to indicate when the program is completed
winsound.Beep(494, 483)
winsound.Beep(392, 483)
winsound.Beep(440, 483)
winsound.Beep(587, 483)
