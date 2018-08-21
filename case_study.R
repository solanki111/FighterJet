
data <- read.csv("C:/Users/../Case_Study/mitbih_train.csv", header=T, na.strings=c(""))
summary(data$BlackBox.R138)
boxplot(data$BlackBox.R138)
pairs(data)
cor(data)

#finding correlation among columns, but it is restricted to 28 cols
library(tibble)
library(dplyr)
library(tidyr)
df <- as.data.frame(data)
df2 <- df %>% 
  as.matrix %>%
  cor %>%
  as.data.frame %>%
  rownames_to_column(var = 'var1') %>%
  gather(var2, value, -var1)

filter(df2, value > .5)

#View missing values
sapply(df,function(x) sum(is.na(x)))
sapply(df, function(x) length(unique(x)))

#complete cases
df.cc <- df[complete.cases(df), ]

#grouping by response levels
df.agg <- aggregate(df, by=list(response=df$BlackBox.R138), data=df, FUN=mean)
View(df.agg)

#Taking the sample for fitting model
library(nnet) 
alpha<-0.01
d <- sort(sample(nrow(df), nrow(df)*alpha))
train <- df[d,]
test <- df[-d,]

# Finding the best subset model
library(leaps)
df.best.subset <- regsubsets(BlackBox.R138 ~ .,
                             data = train,
                             nbest = 10,       # 1 best model for each number of predictors
                             nvmax = 20,    # NULL for no limit on number of variables
                             force.in = NULL, force.out = NULL,
                             method = "exhaustive",
                             really.big=T)
df.best.subset
summary.out <- summary(best.subset)
as.data.frame(summary.out$outmat)
which.max(summary.out$adjr2)
summary.out$which[20,]

#Fitting multinommodel
model <- multinom(BlackBox.R138 ~ ., data=train)
summary(model)
#plot(model)
fitted.results <- predict(model, test, "probs")
#summary(fitted.results)

#fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results < 0.5, 0, 1)
fitted.results

for(i in 1:nrow(fitted.results)) {
  pred.BlackBox.R138[i] <- ifelse(fitted.results[i,1] == 1, "a",
                                  ifelse(fitted.results[i,2] == 1, "b",
                                         ifelse(fitted.results[i,3] == 1, "c",
                                                ifelse(fitted.results[i,4] == 1, "d", "e"))))
}


expected.BlackBox.R138 <- ifelse(test$EG == 0, "a",
                                 ifelse(test$EG == 1, "b",
                                        ifelse(test$EG == 2, "c",
                                               ifelse(test$EG == 3, "d", "e"))))
                                 
tab = table("Expected Value"=expected.BlackBox.R138, 'Predicted value'= pred.BlackBox.R138)
prop.table(tab)
misClasificError <- mean(pred.BlackBox.R138 != expected.BlackBox.R138)
print(paste('Accuracy', (1-mean(pred.BlackBox.R138 != expected.BlackBox.R138))))
                                
library(MASS)
#LDA model
lda.m1 <- lda(BlackBox2138 ~ df.best.subset, data = train)
plot(lda.m1)
