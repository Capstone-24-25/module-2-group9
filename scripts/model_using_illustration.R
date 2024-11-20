# This script is to illustrate the use of models to generate predictions

# For binary class
## Load the model
load("results/binary_model.rds")

## Import the cleaned test data
load("data/clean_test.RData")

## Make the prediction
binary_prediction <- predict(binary_model, claims_test_tfidf_predictor)

# For multi class
## Load the model
load("results/multi_model.rds")

## Make the prediciton
multi_predicition <- predict(multi_model, claims_test_tfidf_predictor)
