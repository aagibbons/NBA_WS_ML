# NBA_WS_ML
Machine Learning project focused on predicting NBA win shares from college statistics.


The main data file used is Data_WS_1_TNT.csv which is the data for the test and training sets, and Project_Test_NBA_Groud_Up_7.py is the python file used to empirically tune each model individually before tuning the weights for the ensemble model.

The data file Data_WS_1_LB.csv is the “lock box” set which was used after the model was tuned as a check for the robustness of the results as these data were not previously used in the calibration of the model. Project_Test_NBA_Groud_Up_8.py uses this “lock box” set to further evaluate the predictions from the model.

An alternate, ongoing approach is to only include players who have a positive value for the Peak Win Shares measure, and using sample averages to account for missing data, which increases the size of the data available. Project_Test_NBA_Groud_Up_9.py is the first code used for this new approach, using the New_Approach_1_TNT.csv file for this analysis.



The files found here were used for the following research paper:

<b>Using Machine Learning to Predict NBA Success</b>

Aidan Gibbons,
University of Toronto

Abstract

The problem of predicting the future performance of college athletes to decide how draft selections should be made in professional sports leagues has been an ongoing issue ever since amateur drafts have first been used. This paper uses machine learning techniques to create an ensemble prediction model to predict the future performance of NBA players drafted out of college. Using college data for players selected in the NBA draft from 1996 to 2012, the ensemble model developed consists of a weighted average of a random forest model, a lasso regression model, and a linear regression model. The order of the predictions obtained from this ensemble model is then compared to how the players were actually selected in the NBA draft. This analysis finds that there is no apparent change in the expected future performance of the players selected using the ensemble predictions compared to real draft selection order. However, the ensemble model does appear to be more risk averse, selecting a worse performing player less than 45% of the time. If NBA decision makers value risk aversion, this model could potentially serve as a better alternative for how college basketball players are selected in the draft.
