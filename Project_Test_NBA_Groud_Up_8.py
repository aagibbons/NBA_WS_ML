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

# start timer to show how long program takes to run
start_time = time.time()

# set a random seed as to allow for the same trees obtained to be used on "lock box" set in future
# also allows for repeatable results
random.seed(0)


# import data into numpy array and list of headings
def get_features():
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

    # return the array of data and list of headings
    return [features, feature_list]


# randomly select years for test set
def rand_years_select(features, feature_list, threshold=0.20):
    # Size of the data set
    data_size = len(features)

    # obtain data on years
    years = features[:, feature_list.index('Draft Year')]
    max_year = int(max(years))
    min_year = int(min(years))

    # record the count for each year in the data
    unique, counts = np.unique(years, return_counts=True)
    years_dict = dict(zip(unique, counts))

    # To ensure threshold for test set size is not too high
    if threshold > 0.5:
        threshold = 0.20
        print('ERROR: Threshold set above 50%, it has been defaulted to 20%')

    # randomly select years for the test set
    test_years = []
    test_set_size = 0
    while test_set_size / data_size < threshold:
        rand = random.randint(min_year, max_year)
        if rand in years and rand not in test_years:
            test_years.append(rand)
            test_set_size += years_dict[rand]

    # sort the years selection
    test_years.sort()

    # return the sorted list of randomly selected years for the test set
    return [test_years, years_dict]


# split data into training set and test set
def split_data(features, feature_list, test_years):

    # determine which row contains draft year data
    year_index = feature_list.index('Draft Year')
    objective_index = feature_list.index('Peak WSPG')

    # split the data into training set and test set
    for y in range(len(test_years)):
        if y == 0:
            test_features = features[(features[:, year_index] == test_years[y]), :]
            train_features = features[(features[:, year_index] != test_years[y]), :]
        else:
            test_features = np.vstack((test_features, features[(features[:, year_index] == test_years[y]), :]))
            train_features = train_features[(train_features[:, year_index] != test_years[y]), :]

    # separate the labels from the training and test sets
    test_labels = np.array(test_features[:, objective_index])
    train_labels = np.array(train_features[:, objective_index])

    # remove the objective variable (labels) and draft year column from features sets and feature list
    train_features = np.delete(train_features, [objective_index, year_index], 1)
    test_features = np.delete(test_features, [objective_index, year_index], 1)
    new_feature_list = feature_list.copy()
    new_feature_list.pop(min(year_index, objective_index))
    new_feature_list.pop(max(year_index, objective_index) - 1)

    # return both the features and labels for the training and test sets
    return [train_features, test_features, train_labels, test_labels, new_feature_list]


# run the random forest model and report the predictions (can also optionally report the importances)
def rand_forest(train_features, test_features, train_labels, new_feature_list, runs=1, n_estimators=1000,
                max_depth=None, min_samples_split=2, max_features="auto", get_importances=False, iteration=0):

    sorted_importances = []

    # run the random forest model a certain number of times (runs) for more accuracy
    for b in range(runs):

        # to track progress of the program
        #print(b)

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
            sorted_importances = np.column_stack((new_feature_list, importances))
            sorted_importances = np.flip(sorted_importances[sorted_importances[:, 1].argsort()], 0)
            print(sorted_importances)

        # toggle to turn off tree figure creation
        export_trees = 0

        # create decision tree pictures, but only for first iteration, first run for the given parameters
        if b == 0 and iteration == 0 and export_trees == 1:
            # Pull out three trees from the forest
            tree1 = rf.estimators_[0]
            tree2 = rf.estimators_[500]
            tree3 = rf.estimators_[999]

            # Export the image to a dot file
            export_graphviz(tree1, out_file='tree1.dot', feature_names=new_feature_list, rounded=True, precision=1)
            export_graphviz(tree2, out_file='tree2.dot', feature_names=new_feature_list, rounded=True, precision=1)
            export_graphviz(tree3, out_file='tree3.dot', feature_names=new_feature_list, rounded=True, precision=1)

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
    print('Coefficients: \n', regr.coef_)

    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(test_labels, predictions))

    lin_reg_r2_score = r2_score(test_labels, predictions)

    return [predictions, lin_reg_r2_score]


# first version of the evaluation procedure. Returns performance of average objective outcome for the first nth picks
def eval_procedure_1(test_labels, test_years, predictions_matrix, years_dict):

    # calculate errors and root mean square error
    if len(np.shape(predictions_matrix)) == 1:
        errors = (predictions_matrix - test_labels) ** 2
    else:
        errors = (np.average(predictions_matrix, axis=1) - test_labels) ** 2
    rmse = (np.average(errors)) ** 0.5
    root_mean_square_error = round(rmse, 4)

    # create a matrix with the actual objective outcomes and the predicted outcomes
    if len(np.shape(predictions_matrix)) == 1:
        objective_actual_predicted = np.column_stack((test_labels, predictions_matrix))
    else:
        objective_actual_predicted = np.column_stack((test_labels, np.mean(predictions_matrix, 1)))

    # obtain locations of start of each draft year to isolate each year
    for b in range(len(test_years)):
        if b == 0:
            div_points = [years_dict[test_years[b]]]
        else:
            div_points.append(div_points[b - 1] + years_dict[test_years[b]])

    # Sorting the predictions to see how the predicted picks would compare to the actual draft selections
    for c in range(len(test_years)):
        if c == 0:
            indiv_year = objective_actual_predicted[:div_points[c], :]
            actual = indiv_year[:min(years_dict.values()), 0]
            predict_check = np.flip(indiv_year[indiv_year[:, 1].argsort()], 0)
            # a check to make sure the predictions were properly sorted
            for d in range(len(predict_check) - 1):
                if predict_check[d + 1, 1] > predict_check[d, 1]:
                    print('ERROR! PROBLEM IN THE SORT FUNCTION!')
                else:
                    continue
            predict = predict_check[:min(years_dict.values()), 0]
            # ========================================
        else:
            indiv_year = objective_actual_predicted[div_points[c - 1]:div_points[c], :]

            # HERE IS WHERE YOU WILL REPEAT THE SAME ANALYSIS
            actual = np.column_stack((actual, indiv_year[:min(years_dict.values()), 0]))
            predict_check = np.flip(indiv_year[indiv_year[:, 1].argsort()], 0)
            # a check to make sure the predictions were properly sorted
            for d in range(len(predict_check) - 1):
                if predict_check[d + 1, 1] > predict_check[d, 1]:
                    print('ERROR! PROBLEM IN THE SORT FUNCTION!')
                else:
                    continue
            predict = np.column_stack((predict, predict_check[:min(years_dict.values()), 0]))
            # ========================================
    evaluation_matrix = np.zeros((min(years_dict.values()), 7))
    if np.shape(actual) != np.shape(predict):
        print('ERROR! PROBLEM WITH THE DIMENSIONS EVALUATION MATRICES!')

    for d in range(np.shape(actual)[0]):
        if np.shape(actual)[1] == 1:
            evaluation_matrix[d, 0] = actual[d, :]
            evaluation_matrix[d, 1] = predict[d, :]
        else:
            evaluation_matrix[d, 0] = np.average(actual[d, :])
            evaluation_matrix[d, 1] = np.average(predict[d, :])
        evaluation_matrix[d, 2] = evaluation_matrix[d, 1] - evaluation_matrix[d, 0]
        for e in range(np.shape(actual)[1]):
            if predict[d, e] > actual[d, e]:
                evaluation_matrix[d, 3] += 1
            if predict[d, e] < actual[d, e]:
                evaluation_matrix[d, 5] += 1
        evaluation_matrix[d, 4] = evaluation_matrix[d, 3] / np.shape(actual[1])
        evaluation_matrix[d, 6] = evaluation_matrix[d, 5] / np.shape(actual[1])

    return [evaluation_matrix, root_mean_square_error]


def rand_forest_eval_1_main(iterations=1, runs=5, max_depth=None, min_samples_split=2, max_features="auto"):
    output_get_features = get_features()
    features = output_get_features[0]
    feature_list = output_get_features[1]
    for t in range(iterations):
        # to track progress of the program
        print('-', t)

        output_rand_years_select = rand_years_select(features, feature_list)
        test_years = output_rand_years_select[0]
        output_split_data = split_data(features, feature_list, test_years)
        train_features = output_split_data[0]
        test_features = output_split_data[1]
        train_labels = output_split_data[2]
        test_labels = output_split_data[3]
        new_feature_list = output_split_data[4]
        output_rand_forest = rand_forest(train_features, test_features, train_labels, new_feature_list, runs,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         max_features=max_features, iteration=t)
        predictions_matrix = output_rand_forest[0]
        years_dict = output_rand_years_select[1]
        output_eval_procedure = eval_procedure_1(test_labels, test_years, predictions_matrix, years_dict)
        evaluation_matrix = output_eval_procedure[0]
        root_mean_square_error = output_eval_procedure[1]
        print(root_mean_square_error)

        if t == 0:
            avg_improvement = evaluation_matrix[:, 2]
            pct_improvement = evaluation_matrix[:, 4]
            pct_worse = evaluation_matrix[:, 6]
            total_root_mean_square_error = np.array(root_mean_square_error)
        else:
            avg_improvement = np.column_stack((avg_improvement, evaluation_matrix[:, 2]))
            pct_improvement = np.column_stack((pct_improvement, evaluation_matrix[:, 4]))
            pct_worse = np.column_stack((pct_worse, evaluation_matrix[:, 6]))
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)

    final_eval = np.zeros((len(avg_improvement), 3))
    for u in range(np.shape(avg_improvement)[0]):
        if len(np.shape(avg_improvement)) == 1:
            final_eval[u, 0] = avg_improvement[u]
            final_eval[u, 1] = pct_improvement[u]
            final_eval[u, 2] = pct_worse[u]
        else:
            final_eval[u, 0] = np.average(avg_improvement[u, :])
            final_eval[u, 1] = np.average(pct_improvement[u, :])
            final_eval[u, 2] = np.average(pct_worse[u, :])
    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
    print(final_eval)
    print(avg_total_root_mean_square_error)

    return [final_eval, avg_total_root_mean_square_error]


def lasso_eval_1_main(iterations=1, runs=5, alpha=1.0, tol=0.0001):
    output_get_features = get_features()
    features = output_get_features[0]
    feature_list = output_get_features[1]
    for t in range(iterations):

        output_rand_years_select = rand_years_select(features, feature_list)
        test_years = output_rand_years_select[0]
        output_split_data = split_data(features, feature_list, test_years)
        train_features = output_split_data[0]
        test_features = output_split_data[1]
        train_labels = output_split_data[2]
        test_labels = output_split_data[3]

        # ========
        output_lasso = lasso(train_features, test_features, train_labels, test_labels, runs=runs, alpha=alpha, tol=tol)
        predictions_matrix = output_lasso[0]
        lasso_score_avg = output_lasso[1]
        # =====

        years_dict = output_rand_years_select[1]
        output_eval_procedure = eval_procedure_1(test_labels, test_years, predictions_matrix, years_dict)
        evaluation_matrix = output_eval_procedure[0]
        root_mean_square_error = output_eval_procedure[1]

        if t == 0:
            avg_improvement = evaluation_matrix[:, 2]
            pct_improvement = evaluation_matrix[:, 4]
            pct_worse = evaluation_matrix[:, 6]
            total_root_mean_square_error = np.array(root_mean_square_error)
            total_lasso_score_avg = np.array(lasso_score_avg)
        else:
            avg_improvement = np.column_stack((avg_improvement, evaluation_matrix[:, 2]))
            pct_improvement = np.column_stack((pct_improvement, evaluation_matrix[:, 4]))
            pct_worse = np.column_stack((pct_worse, evaluation_matrix[:, 6]))
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)
            total_lasso_score_avg = np.append(total_lasso_score_avg, lasso_score_avg)

    final_eval = np.zeros((len(avg_improvement), 3))
    for u in range(np.shape(avg_improvement)[0]):
        if len(np.shape(avg_improvement)) == 1:
            final_eval[u, 0] = avg_improvement[u]
            final_eval[u, 1] = pct_improvement[u]
            final_eval[u, 2] = pct_worse[u]
        else:
            final_eval[u, 0] = np.average(avg_improvement[u, :])
            final_eval[u, 1] = np.average(pct_improvement[u, :])
            final_eval[u, 2] = np.average(pct_worse[u, :])
    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
        avg_total_lasso_score_avg = total_lasso_score_avg
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
        avg_total_lasso_score_avg = round(np.average(total_lasso_score_avg), 4)
    print(final_eval)
    print(avg_total_root_mean_square_error)
    print(avg_total_lasso_score_avg)
    print('MAX', max(total_lasso_score_avg))
    print('MIN', min(total_lasso_score_avg))

    return [final_eval, avg_total_root_mean_square_error, avg_total_lasso_score_avg]


def lin_reg_eval_1_main(iterations=1):
    output_get_features = get_features()
    features = output_get_features[0]
    feature_list = output_get_features[1]
    for t in range(iterations):

        output_rand_years_select = rand_years_select(features, feature_list)
        test_years = output_rand_years_select[0]
        output_split_data = split_data(features, feature_list, test_years)
        train_features = output_split_data[0]
        test_features = output_split_data[1]
        train_labels = output_split_data[2]
        test_labels = output_split_data[3]

        # ========
        output_lin_reg = lin_reg(train_features, test_features, train_labels, test_labels)
        predictions = output_lin_reg[0]
        lin_reg_r2_score = output_lin_reg[1]
        # ========

        years_dict = output_rand_years_select[1]
        output_eval_procedure = eval_procedure_1(test_labels, test_years, predictions, years_dict)
        evaluation_matrix = output_eval_procedure[0]
        root_mean_square_error = output_eval_procedure[1]

        if t == 0:
            avg_improvement = evaluation_matrix[:, 2]
            pct_improvement = evaluation_matrix[:, 4]
            pct_worse = evaluation_matrix[:, 6]
            total_root_mean_square_error = np.array(root_mean_square_error)
            total_lin_reg_r2_score = np.array(lin_reg_r2_score)
        else:
            avg_improvement = np.column_stack((avg_improvement, evaluation_matrix[:, 2]))
            pct_improvement = np.column_stack((pct_improvement, evaluation_matrix[:, 4]))
            pct_worse = np.column_stack((pct_worse, evaluation_matrix[:, 6]))
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)
            total_lin_reg_r2_score = np.append(total_lin_reg_r2_score, lin_reg_r2_score)
    print('MAX', max(total_lin_reg_r2_score))
    print('MIN', min(total_lin_reg_r2_score))
    final_eval = np.zeros((len(avg_improvement), 3))
    for u in range(np.shape(avg_improvement)[0]):
        if len(np.shape(avg_improvement)) == 1:
            final_eval[u, 0] = avg_improvement[u]
            final_eval[u, 1] = pct_improvement[u]
            final_eval[u, 2] = pct_worse[u]
        else:
            final_eval[u, 0] = np.average(avg_improvement[u, :])
            final_eval[u, 1] = np.average(pct_improvement[u, :])
            final_eval[u, 2] = np.average(pct_worse[u, :])
    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
        avg_total_lin_reg_r2_score = total_lin_reg_r2_score
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
        avg_total_lin_reg_r2_score = round(np.average(total_lin_reg_r2_score), 4)
    print(final_eval)
    print(avg_total_root_mean_square_error)
    print(avg_total_lin_reg_r2_score)

    return [final_eval, avg_total_root_mean_square_error, avg_total_lin_reg_r2_score]


def ensemble_eval_1_main(iterations=1, runs_rf=5, max_depth=None, min_samples_split=2, max_features="auto",
                         runs_las=5, alpha=1.0, tol=0.0001, weight_rf=1, weight_las=1, weight_lr=1):
    # I WOULD DO THIS DIFFERENTLY IF GIVEN MORE TIME, IF YOU MAKE THESE 3 LINES INTO IT'S OWN FUNCTION,
    # YOU COULD PROBABLY REUSE THE PREVIOUS FUNCTIONS FOR THE DIFFERENT MODELS
    output_get_features = get_features()
    features = output_get_features[0]
    feature_list = output_get_features[1]
    for t in range(iterations):
        print(t, '/', iterations)

        output_rand_years_select = rand_years_select(features, feature_list)
        test_years = output_rand_years_select[0]
        output_split_data = split_data(features, feature_list, test_years)
        train_features = output_split_data[0]
        test_features = output_split_data[1]
        train_labels = output_split_data[2]
        test_labels = output_split_data[3]
        new_feature_list = output_split_data[4]
        years_dict = output_rand_years_select[1]

        # ==================================================
        # create lockbox matrix
        lockbox = pd.read_csv('Data_WS_1_LB.csv')

        lockbox = lockbox.drop('Name', axis=1)
        lockbox = lockbox.drop('Pick', axis=1)

        lockbox = pd.get_dummies(lockbox)

        lockbox_labels = np.array(lockbox)
        lockbox_years = lockbox_labels[:, 1]
        lockbox_labels = lockbox_labels[:, 0]

        unique, counts = np.unique(lockbox_years, return_counts=True)
        lockbox_years_dict = dict(zip(unique, counts))

        lockbox_years = [2011, 2012]

        lockbox = lockbox.drop('Peak WSPG', axis=1)
        lockbox = lockbox.drop('Draft Year', axis=1)

        lockbox_features = np.array(lockbox)
        # ==============================================================

        # obtain random forest output
        output_rand_forest = rand_forest(train_features, lockbox_features, train_labels, new_feature_list, runs=runs_rf,
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
        output_lasso = lasso(train_features, lockbox_features, train_labels, lockbox_labels, runs=runs_las, alpha=alpha,
                             tol=tol)
        # obtain lasso predictions matrix
        predictions_matrix_las = output_lasso[0]
        # collapse the prediction matrix into the averages
        if len(np.shape(predictions_matrix_las)) == 1:
            predictions_las = predictions_matrix_las
        else:
            predictions_las = np.average(predictions_matrix_las, axis=1)

        # obtain linear regression output
        output_lin_reg = lin_reg(train_features, lockbox_features, train_labels, lockbox_labels)
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
        #output_eval_procedure = eval_procedure_1(test_labels, test_years, predictions_en, years_dict)
        output_eval_procedure = eval_procedure_1(lockbox_labels, lockbox_years, predictions_en, lockbox_years_dict)
        evaluation_matrix = output_eval_procedure[0]
        root_mean_square_error = output_eval_procedure[1]

        print(root_mean_square_error)

        # create final evaluation and average total root mean square error outputs
        if t == 0:
            avg_improvement = evaluation_matrix[:, 2]
            pct_improvement = evaluation_matrix[:, 4]
            pct_worse = evaluation_matrix[:, 6]
            total_root_mean_square_error = np.array(root_mean_square_error)
        else:
            avg_improvement = np.column_stack((avg_improvement, evaluation_matrix[:, 2]))
            pct_improvement = np.column_stack((pct_improvement, evaluation_matrix[:, 4]))
            pct_worse = np.column_stack((pct_worse, evaluation_matrix[:, 6]))
            total_root_mean_square_error = np.append(total_root_mean_square_error, root_mean_square_error)
    final_eval = np.zeros((len(avg_improvement), 3))
    for u in range(np.shape(avg_improvement)[0]):
        if len(np.shape(avg_improvement)) == 1:
            final_eval[u, 0] = avg_improvement[u]
            final_eval[u, 1] = pct_improvement[u]
            final_eval[u, 2] = pct_worse[u]
        else:
            final_eval[u, 0] = np.average(avg_improvement[u, :])
            final_eval[u, 1] = np.average(pct_improvement[u, :])
            final_eval[u, 2] = np.average(pct_worse[u, :])
    if iterations == 1:
        avg_total_root_mean_square_error = total_root_mean_square_error
    else:
        avg_total_root_mean_square_error = round(np.average(total_root_mean_square_error), 4)
    print(final_eval)
    print(avg_total_root_mean_square_error)

    return [final_eval, avg_total_root_mean_square_error]


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
    output = rand_forest_eval_1_main(iterations, runs, max_depth, min_samples_split, max_features)
    improvement_matrix = output[0]
    avg_total_root_mean_square_error = output[1]

    # creating file name then exporting results as .csv file
    filename = 'Ver-4.0_md-' + str(max_depth) + '_mss-' + str(min_samples_split) + '_mf-' + str(max_features) + \
               '_itr-' + str(iterations) + '_run-' + str(runs) + '_rmse-' + str(avg_total_root_mean_square_error) + \
               '.csv'

    export = pd.DataFrame(improvement_matrix, columns=['avg_improvement', 'pct_improvement', 'pct_worse'])
    # export.to_csv(filename)

# run lasso regression model
if las == 1 and rf == 0 and lr == 0:
    # parameters used (lasso)
    iterations = 100
    runs = 5
    alpha = 0.13  # options: float. WARNING: do not set this to 0
    tol = 0.000001

    # running the output
    output = lasso_eval_1_main(iterations, runs, alpha, tol)
    improvement_matrix = output[0]
    avg_total_root_mean_square_error = output[1]
    avg_total_lasso_score_avg = output[2]

    # creating file name then exporting results as .csv file
    filename = 'Ver-L7.0_alpha-' + str(alpha) + '_tol-' + str(tol) + '_itr-' + str(iterations) + '_run-' + str(runs) + \
               '_score-' + str(avg_total_lasso_score_avg) + '_rmse-' + str(avg_total_root_mean_square_error) + '.csv'

    #export = pd.DataFrame(improvement_matrix, columns=['avg_improvement', 'pct_improvement', 'pct_worse'])
    # export.to_csv(filename)

# run linear regression
if lr == 1 and las == 0 and rf == 0:
    # parameters used (linear regression)
    iterations = 100

    # running the output
    output = lin_reg_eval_1_main(iterations)
    improvement_matrix = output[0]
    avg_total_root_mean_square_error = output[1]
    avg_total_lin_reg_r2_score = output[2]

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
    improvement_matrix = output[0]
    avg_total_root_mean_square_error = output[1]

    # creating file name then exporting results as .csv file
    filename = 'Ver-E8.0_iter-' + str(iterations) + '_runs_rf-' + str(runs_rf) + '_md-' + str(max_depth) + '_mss-' + \
               str(min_samples_split) + '_mf-' + str(max_features) + '_runs_las-' + str(runs_las) + '_alpha-' + \
               str(alpha) + '_tol-' + str(tol) + '_w_rf-' + str(weight_rf) + '_w_las-' + str(weight_las) + '_w_lr-' + \
               str(weight_lr) + '_rmse-' + str(avg_total_root_mean_square_error) + '.csv'

    export = pd.DataFrame(improvement_matrix, columns=['avg_improvement', 'pct_improvement', 'pct_worse'])
    export.to_csv(filename)

# print result of how long program takes to run
print()
print("My program took", time.time() - start_time, "to run")

# sound to indicate when the program is completed
winsound.Beep(494, 483)
winsound.Beep(392, 483)
winsound.Beep(440, 483)
winsound.Beep(587, 483)
