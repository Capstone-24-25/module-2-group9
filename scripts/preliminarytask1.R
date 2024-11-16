# packages
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

# path to activity files on repo
# url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
#source(paste(url, 'projection-functions.R', sep = ''))

# read in data
#claims <- paste(url, 'claims-multi-tfidf.csv', sep = '') %>%
#  read_csv()


# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')



## WITHOUT HEADERS
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
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

#turn into tf-idf
no_headers_tfidf <- claims_clean %>% 
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words',
                stopwords = str_remove_all(stop_words$word, 
                                           '[[:punct:]]')) %>%
  mutate(token.lem = lemmatize_words(token)) %>%
  filter(str_length(token.lem) > 2) %>%
  count(.id, bclass, mclass, token.lem, name = 'n') %>%
  bind_tf_idf(term = token.lem, 
              document = .id,
              n = n) %>%
  pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
              names_from = 'token.lem',
              values_from = 'tf_idf',
              values_fill = 0)


# partition data
set.seed(110122)
partitions1 <- no_headers_tfidf %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm1 <- testing(partitions1) %>%
  select(-.id, -bclass, -mclass)
test_labels1 <- testing(partitions1) %>%
  select(.id, bclass, mclass)

# same, training set
train_dtm1 <- training(partitions1) %>%
  select(-.id, -bclass, -mclass)
train_labels1 <- training(partitions1) %>%
  select(.id, bclass, mclass)

# find projections based on training data
proj_out1 <- projection_fn(.dtm = train_dtm1, .prop = 0.7)
train_dtm_projected1 <- proj_out1$data

# how many components were used?
proj_out1$n_pc

#regression
train1 <- train_labels1 %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected1)


fit1 <- glm(bclass~., data = train1, family = binomial)

#project test data
test_dtm_projected1 <- reproject_fn(.dtm = test_dtm1, proj_out1)

#get predictions
preds1 <- predict(fit1,
                  newdata = as.data.frame(test_dtm_projected1),
                  type = 'response')

#test-labels with predictions
pred_df1 <- test_labels1 %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds1)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# Calculate metrics
pred_df1 %>% class_metrics(truth = bclass, 
                           estimate = bclass.pred, 
                           pred, 
                           event_level = 'second')









# WITH HEADERS
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1') %>%
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
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}


claims_clean_headers <- claims_raw %>%
  parse_data()

#load('data/claims-clean-headers.RData')
headers_clean <- claims_clean_headers %>%
  select(-c(1:5), -7)

headers_tfidf <- headers_clean %>% 
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words',
                stopwords = str_remove_all(stop_words$word, 
                                           '[[:punct:]]')) %>%
  mutate(token.lem = lemmatize_words(token)) %>%
  filter(str_length(token.lem) > 2) %>%
  count(.id, bclass, mclass, token.lem, name = 'n') %>%
  bind_tf_idf(term = token.lem, 
              document = .id,
              n = n) %>%
  pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
              names_from = 'token.lem',
              values_from = 'tf_idf',
              values_fill = 0)


# Step 1: Partitioning
# partition data
set.seed(110122)
partitions <- headers_tfidf %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)


# PCA with header data

# Step 2: Projection
# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc


# fit header data into logistic regression
# Step 3: Regression
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)


fit <- glm(bclass~., data = train, family = binomial)

#project test data
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

#get predictions
preds <- predict(fit,
                 newdata = as.data.frame(test_dtm_projected),
                 type = 'response')

#test-labels with predictions
pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# Evaluate errors on test set
class_metrics <- metric_set(sensitivity, 
                            specificity, 
                            accuracy,
                            roc_auc)

# Calculate metrics
pred_df %>% class_metrics(truth = bclass, 
                          estimate = bclass.pred, 
                          pred, 
                          event_level = 'second')

