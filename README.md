# Data-analysis-and-machine-learning
Blood pressure analysis

Projects done between November 2022 and January 2023

### Project 1:
File: "project 1.py"
 - Prepare statistics for each patient,
 - visualizations: pointcare charts, histograms, series visuals
 - signal analysis and differential signal analysis in slip windows,
 - stationarity evaluation,
 - calculation of SDNN, RMSSD (the square root of the mean of the sum of the squares of differences between adjacent NN intervals), pNN50, pNN20,
 - checking whether the heart slows down or speeds up,
 - calculating the probability of whether the heart slows down, speeds up or does not change (one, two and three element),
 - finding quantum patterns,
 - finding window with max mean and max std,
 - presentation of results in tables
 
 ### Project 2:
 File: "project 2.py"
 I worked on the results I got from project 1 ("wyniki.csv", "okna_sr.csv", "okna_std.csv").
 - Summary of numeric attributes,
 - data histograms,
 - splitting the data into training and testing sets using stratified sampling,
 - working on first model - main table "wyniki":
   - data analysis:
      - copy of training set,
      - finding correlations,
      - correlation graph,
   - data preparation:
      - separation of labels from data,
      - division of attributes into categorical and numerical,
      - data transformation using MinMaxScaler and OrdinalEncoder,
      - PCA, explaining 95% of the variance, plots,
   - model selection and training:
      - defines a function to compare errors,
      - finding best hyperparameters for SGDRegressor with GridSearchCV,
      - cross validation test for chosen parameters,
      - evaluation using the TEST set,
      - calculation of the confidence interval
- repeating the above steps for the tables "okna_sr", "okna_std"
- comparison of errors for 3 models
    
Longer polish description: file "project2.docx"
 
