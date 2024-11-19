set.seed(110122)

# tokenize into bigrams
headers_bigrams <- claims_clean_headers %>%
  select(.id, bclass, text_clean) %>%
  unnest_tokens(output = bigram, 
                input = text_clean, 
                token = 'ngrams', 
                n = 2, 
                stopwords = str_remove_all(stop_words$word, '[[:punct:]]'))

# count bigrams and compute TF-IDF
headers_bigrams_tfidf <- headers_bigrams %>%
  count(.id, bclass, bigram, name = 'n') %>%
  bind_tf_idf(term = bigram, 
              document = .id, 
              n = n) %>%
  filter(n>=5) %>% 
  pivot_wider(id_cols = c(.id, bclass),
              names_from = bigram,
              values_from = tf_idf,
              values_fill = 0)

# partition data
partitions_bigrams <- headers_bigrams_tfidf %>% initial_split(prop = 0.8)

train_dtm_bigrams <- training(partitions_bigrams) %>%
  select(-.id, -bclass)
train_labels_bigrams <- training(partitions_bigrams) %>%
  select(.id, bclass)

test_dtm_bigrams <- testing(partitions_bigrams) %>%
  select(-.id, -bclass)
test_labels_bigrams <- testing(partitions_bigrams) %>%
  select(.id, bclass)

# first logistic regression
# PCA projection for bigram data
train_dtm_bigrams_sparse <- train_dtm_bigrams %>%
  as.matrix() %>%
  as('sparseMatrix') 
svd_out_bigrams <- sparsesvd(train_dtm_bigrams_sparse)

# projected data frame
train_dtm_projected2 <- svd_out_bigrams$u %*% diag(svd_out_bigrams$d)

# assign column names
colnames(train_dtm_projected2) <- paste0("PC", 1:ncol(train_dtm_projected2))

# regression
train2 <- train_labels_bigrams %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected2)

fit2 <- glm(bclass~., data = train2, family = binomial)

# re-projection
reproject_fn1 <- function(.dtm, train_projected) {
  .dtm_sparse <- as(.dtm, "sparseMatrix")
  test_projected <- as.matrix(.dtm_sparse %*% train_projected$v %*% diag(1 / train_projected$d))
  colnames(test_projected) <- paste0("PC", 1:ncol(test_projected))
  return(test_projected)
}

# project test data
test_dtm_projected2 <- reproject_fn1(.dtm = test_dtm_bigrams, svd_out_bigrams)

# get predictions
preds2 <- predict(fit2,
                  newdata = as.data.frame(test_dtm_projected2),
                  type = 'link')

test_data_combined <- as.data.frame(test_dtm_projected2) %>%
  mutate(log_odds = preds2) # Add log-odds as a feature

# fit second logistic regression 
train_combined <- train_labels_bigrams %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected2, log_odds = predict(fit2, type = 'link'))

fit_combined <- glm(bclass ~ ., data = train_combined, family = binomial)

test_preds_combined <- predict(fit_combined, 
                               newdata = test_data_combined, 
                               type = 'response')

pred_df_combined <- test_labels_bigrams %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(test_preds_combined)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

metrics2 <- pred_df_combined %>% class_metrics(truth = bclass, 
                                       estimate = bclass.pred, 
                                       pred, 
                                       event_level = 'second')

metrics2
