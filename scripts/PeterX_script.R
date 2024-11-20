## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse) 
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  words_df <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n')
  
  bigrams_df <- parse_data.out %>%
    unnest_tokens(output = bigram, 
                  input = text_clean, 
                  token = 'ngrams', n = 2) %>%
    count(.id, bclass, bigram, name = 'n')
  
  combined_df <- bind_rows(words_df, bigrams_df)

  tidy_df <- combined_df %>%
    bind_tf_idf(term = c(token.lem, bigram), 
                document = .id, 
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = c(token.lem, bigram),
                values_from = 'tf_idf',
                values_fill = 0)
  
  return(bigrams_df, tidy_df)
}


## PREPROCESSING
#################

# can comment entire section out if no changes to preprocessing.R
source('module-2-group9/scripts/preprocessing.R')

# load raw data
load('module-2-group9/data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()
tokenized_data <- nlp_fn(claims_clean)
tokenized_data
bigrams_df <- claims_clean %>%
  unnest_tokens(output = bigram, 
                input = text_clean, 
                token = 'ngrams', n = 2) %>%
  count(.id, bclass, bigram, name = 'n') %>%
  bind_tf_idf(term = c(token.lem, bigram), 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c('.id', 'bclass'),
              names_from = c(token.lem, bigram),
              values_from = 'tf_idf',
              values_fill = 0)
bigrams_df
# Prepare data for PCA and regression
X <- select(tokenized_data, -c(.id, bclass))
Y <- tokenized_data$bclass

# PCA on word token data
pca_words <- prcomp(X, scale. = TRUE)

# Fit logistic regression on principal components
fit_logistic1 <- glmnet(pca_words$x, Y, family = 'binomial')

# Predict using the logistic model
predicted_scores <- predict(fit_logistic1, newx = pca_words$x, type = 'response')

# Prepare Bigrams, Perform PCA, and Fit the Second Logistic Regression
# (Assuming bigram preparation and extraction have been handled similarly)
X_bigrams <- select(bigrams_df, -c(.id, bclass))
pca_bigrams <- prcomp(X_bigrams, scale. = TRUE)

input_data <- cbind(predicted_scores, pca_bigrams$x)

fit_logistic2 <- glmnet(input_data, Y, family = 'binomial')

# Validation and Testing
train_indices <- createDataPartition(Y, p = 0.8, list = FALSE)
train_data <- input_data[train_indices, ]
train_labels <- Y[train_indices]

test_data <- input_data[-train_indices, ]
test_labels <- Y[-train_indices]

predictions <- predict(fit_logistic2, newx = test_data, type = "response")
performance <- confusionMatrix(as.factor(predictions > 0.5), as.factor(test_labels))
print(performance)





## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(caret)
library(glmnet)

# load cleaned data


# partition
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# If having library conflicts
# install.packages("keras", type = "source")
# library(keras)
# install_keras()

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)

# define NN architecture
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

# configure for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# train
history <- model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 5)

## CHECK TEST SET ACCURACY HERE
model$weights

test_text <- testing(partitions) %>% 
  pull(text_clean)

test_labels <- testing(partitions) %>% 
  pull(bclass) %>% 
  as.numeric() -1

evaluation <- evaluate(model, test_text, test_labels) # 0.775/0.785

# save the entire model as a SavedModel
save_model_tf(model, "results/example-model")



require(tidyverse)
require(keras)
require(tensorflow)
load('module-2-group9/data/claims-test.RData')
load('module-2-group9/data/claims-raw.RData')
source('module-2-group9/scripts/preprocessing.R')
tf_model <- load_model_tf('module-2-group9/results/example-model')

# apply preprocessing pipeline
clean_df <- claims_test %>%
  slice(1:100) %>%
  parse_data() %>%
  select(.id, text_clean)

# grab input
x <- clean_df %>%
  pull(text_clean)

# compute predictions
preds <- predict(tf_model, x) %>%
  as.numeric()

class_labels <- claims_raw %>% pull(bclass) %>% levels()

pred_classes <- factor(preds > 0.5, labels = class_labels)

# export (KEEP THIS FORMAT IDENTICAL)
pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes) %>%
  select(.id, bclass.pred)

save(pred_df, file = 'results/example-preds.RData')
