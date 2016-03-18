# Kaggle-BNP-Paribas
Kaggle BNP Paribas XGB code
setwd("D:/Kaggle/BNP Paribas")

train_raw = read.csv("train.csv")
str(train_raw)
summary(train_raw)
print(dim(train_raw))
print(sapply(train_raw, class))

y <- train_raw$target
train_raw$target <- NULL
train_raw$ID <- NULL
n <- nrow(train_raw)




test_raw = read.csv("test.csv")
str(test_raw)
summary(test_raw)
test_raw$ID <- NULL


all_data = rbind(train_raw,test_raw )

N <- ncol(all_data)
all_data$NACount_N <- rowSums(is.na(all_data)) / N 

feature.names <- names(all_data)

all_data$ZeroCount <- rowSums(all_data[,feature.names]== 0) / N

highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
                      "v51","v53","v54","v63","v73","v81",
                      "v82","v89","v92","v95","v105","v107",
                      "v108","v109","v116","v117","v118",
                      "v119","v123","v124","v128")
all_data <- all_data[,-which(names(all_data) %in% highCorrRemovals)]

for (f in feature.names) {
  if (class(all_data[[f]])=="character" || class(all_data[[f]])=="factor") {
    all_data[[f]] <- as.integer(factor(all_data[[f]]))
  }
}


train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 
y <- train_raw$target

tmpJ = 1:ncol(test)
ksMat = NULL
for (j in tmpJ) {
  cat(j," ")
  ksMat = rbind(ksMat, cbind(j, ks.test(train[,j],test[,j])$statistic))
}

ksMat2 = ksMat[ksMat[,2]<0.007,]
feats = as.numeric(ksMat2[,1]) 
cat(length(feats),"\n")
cat(names(train)[feats],"\n")
var_to_drop <- setdiff(names(all_data), names(train)[feats])
cat("\nVars to drop:", var_to_drop, "\n")
all_data <- rbind(train[,feats],test[,feats])

na.roughfix2 <- function (object, ...) {
  res <- lapply(object, roughfix)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x) {
  missing <- is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    x[missing] <- median.default(x[!missing])
  } else if (is.factor(x)) {
    freq <- table(x)
    x[missing] <- names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

all_data <- na.roughfix2(all_data)

train <- all_data[1:n,]
test <- all_data[(n+1):nrow(all_data),] 

library(xgboost)
xgtrain = xgb.DMatrix(data.matrix(train), label = y, missing=NA)
xgtest = xgb.DMatrix(data.matrix(test), missing=NA)

docv <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 2
    , data = xgtrain
    , early.stop.round = 10
    , maximize = FALSE
    , nthread = 8
  )
  gc()
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "logloss"
  , "eta" = 0.05
  , "subsample" = 0.9
  , "colsample_bytree" = 0.9
  , "min_child_weight" = 1
  , "max_depth" = 10
  , "gamma" = 0.1
)


cat("Training a XGBoost classifier with cross-validation\n")
set.seed(2018)
cv <- docv(param0, 500) 


ensemble <- rep(0, nrow(test))

cv <- round(cv * 1.5)
cat("Calculated rounds:", cv, " Starting ensemble\n")

doTest <- function(param0, iter) {
  watchlist <- list('train' = xgtrain)
  model = xgb.train(
    nrounds = iter
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 20
    , nthread = 8
  )
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}

for (i in 1:5) {
  print(i)
  set.seed(i + 2017)
  p <- doTest(param0, cv) 
  # use 40% to 50% more than the best iter rounds from your cross-fold number.
  # as you have another 50% training data now, which gives longer optimal training time
  ensemble <- ensemble + p
}

submission <- read.csv("sample_submission.csv")
submission$PredictedProb <- ensemble/i
write.csv(submission, "bnp-xgb_without_ensemble.csv", row.names=F, quote=F)
summary(submission$PredictedProb)
