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
    while test_set_size/data_size < threshold:
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
    new_feature_list.pop(max(year_index, objective_index)-1)

    # return both the features and labels for the training and test sets
    return [train_features, test_features, train_labels, test_labels, new_feature_list]


# run the random forest model and report the predictions (can also optionally report the importances)
def rand_forest(train_features, test_features, train_labels, new_feature_list, runs=1, n_estimators=1000,
                max_depth=None, max_features="auto", get_importances=False):
    
    sorted_importances = []
    
    # run the random forest model a certain number of times (runs) for more accuracy
    for b in range(runs):

        # to track progress of the program
        print(b)

        # Instantiate model with chosen parameters
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                   random_state=b)
        
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

    return [predictions_matrix, sorted_importances]


# first version of the evaluation procedure. Returns performance of average objective outcome for the first nth picks
def eval_procedure_1(test_labels, test_years, predictions_matrix, years_dict):

    # calculate errors and mean absolute error
    errors = abs(np.average(predictions_matrix, axis=1) - test_labels)
    mean_abs_error = round(np.mean(errors), 4)

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
            div_points.append(div_points[b-1]+years_dict[test_years[b]])

    # EXPLANATION NEEDED
    for c in range(len(test_years)):
        if c == 0:
            indiv_year = objective_actual_predicted[:div_points[c], :]
            # HERE IS WHERE YOU WILL DO AN ANALYSIS
            actual = indiv_year[:min(years_dict.values()), 0]
            predict_check = np.flip(indiv_year[indiv_year[:, 1].argsort()], 0)
            # a check to make sure the predictions were properly sorted
            for d in range(len(predict_check)-1):
                if predict_check[d+1, 1] > predict_check[d, 1]:
                    print('ERROR! PROBLEM IN THE SORT FUNCTION!')
                else:
                    continue
            predict = predict_check[:min(years_dict.values()), 0]
            # ========================================
        else:
            indiv_year = objective_actual_predicted[div_points[c-1]:div_points[c], :]

            # HERE IS WHERE YOU WILL REPEAT THE SAME ANALYSIS
            actual = np.column_stack((actual, indiv_year[:min(years_dict.values()), 0]))
            predict_check = np.flip(indiv_year[indiv_year[:, 1].argsort()], 0)
            # a check to make sure the predictions were properly sorted
            for d in range(len(predict_check)-1):
                if predict_check[d+1, 1] > predict_check[d, 1]:
                    print('ERROR! PROBLEM IN THE SORT FUNCTION!')
                else:
                    continue
            predict = np.column_stack((predict, predict_check[:min(years_dict.values()), 0]))
            # ========================================
    evaluation_matrix = np.zeros((min(years_dict.values()), 5))
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
        evaluation_matrix[d, 4] = evaluation_matrix[d, 3] / np.shape(actual[1])
    return [evaluation_matrix, mean_abs_error]


def rand_forest_eval_1_main(iterations=1, runs=5, max_depth=None, max_features="auto"):

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
                                         max_depth=max_depth, max_features=max_features)
        predictions_matrix = output_rand_forest[0]
        years_dict = output_rand_years_select[1]
        output_eval_procedure = eval_procedure_1(test_labels, test_years, predictions_matrix, years_dict)
        evaluation_matrix = output_eval_procedure[0]
        mean_abs_error = output_eval_procedure[1]
        print(mean_abs_error)

        if t == 0:
            avg_improvement = evaluation_matrix[:, 2]
            pct_improvement = evaluation_matrix[:, 4]
            total_mean_absolute_error = np.array(mean_abs_error)
        else:
            avg_improvement = np.column_stack((avg_improvement, evaluation_matrix[:, 2]))
            pct_improvement = np.column_stack((pct_improvement, evaluation_matrix[:, 4]))
            total_mean_absolute_error = np.append(total_mean_absolute_error, mean_abs_error)

    final_eval = np.zeros((len(avg_improvement), 2))
    for u in range(np.shape(avg_improvement)[0]):
        if len(np.shape(avg_improvement)) == 1:
            final_eval[u, 0] = avg_improvement[u]
            final_eval[u, 1] = pct_improvement[u]
        else:
            final_eval[u, 0] = np.average(avg_improvement[u, :])
            final_eval[u, 1] = np.average(pct_improvement[u, :])
    if iterations == 1:
        avg_total_mse = total_mean_absolute_error
    else:
        avg_total_mse = round(np.mean(total_mean_absolute_error), 4)
    print(final_eval)
    print(avg_total_mse)
    return [final_eval, avg_total_mse]


for md in range(2, 6):
    # parameters used
    iterations = 100
    runs = 5
    max_depth = md
    max_features = "sqrt"  # options: auto(default, =n_features), sqrt, log2, int, float(fraction)

    # running the output
    output = rand_forest_eval_1_main(iterations, runs, max_depth, max_features)
    improvement_matrix = output[0]
    avg_total_mse = output[1]

    # creating file name then exporting results as .csv file
    filename = 'Ver-3.0_md-'+str(max_depth)+'_mf-'+str(max_features)+'_itr-'+str(iterations)+'_run-'+str(runs) +\
               '_mae-'+str(avg_total_mse)+'.csv'

    export = pd.DataFrame(improvement_matrix, columns=['avg_improvement', 'pct_improvement'])
    export.to_csv(filename)

# print result of how long program takes to run
print()
print("My program took", time.time() - start_time, "to run")

# sound to indicate when the program is completed
winsound.Beep(494, 483)
winsound.Beep(392, 483)
winsound.Beep(440, 483)
winsound.Beep(587, 483)
