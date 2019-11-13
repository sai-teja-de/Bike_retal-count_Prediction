#Cleaning the environment
rm(list = ls())

#setting working directory
setwd("E:/Data Science/EDWISOR/Project_1")
getwd()

#loading the libraries 


L = c('tidyr','ggplot2','corrgram','usdm','caret','DMwR','rpart','randomForest')

#loading packages
lapply(L, require, character.only = TRUE)
rm(L)

#Loading Dataset

b_rent = read.csv('day.csv',header = TRUE)

                                       ### DATA EXPLORATION  ###
# Structure of data
str(b_rent)

# Summary of data
summary(b_rent)

head(b_rent)

#changing the column names
library(data.table)
setnames(b_rent,old =c('dteday','yr','mnth','weathersit','hum','cnt'), new =c('date','year','month','weather','humdity','count'))

#checking column names
colnames(b_rent)

#Changing the data types of few variables

b_rent$date =as.Date(as.character(b_rent$date))

cat_names =c('season','year','month','holiday','weekday','workingday','weather')
for (i in cat_names) {
  print(i)
  b_rent[,i]=as.factor(b_rent[,i])
}

                                    #### MISSING VALUE ANALYSIS #####

missing_values=data.frame(apply(b_rent, 2, function(x){sum(is.na(x))}))
View(missing_values)
#we have no missing values in the dataset

                                     ##### OUTLIER ANALYSIS #####
numeric_index= sapply(b_rent, is.numeric)
numeric_data = b_rent[,numeric_index]
num_cnames = colnames(numeric_data)

for (i in 1:length(num_cnames))
{
  assign(paste0("bp",i), ggplot(aes_string(y = (num_cnames[i]), x = "count"), data = subset(b_rent))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "white" ,outlier.shape=15,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=num_cnames[i],x="count")+
           ggtitle(paste("Box plot of count for",num_cnames[i])))
}


# ## Plotting plots together
gridExtra::grid.arrange(bp1,bp2,ncol=2)
gridExtra::grid.arrange(bp4,bp5,ncol=2)
gridExtra::grid.arrange(bp6,bp3,ncol=2)

# From the plots we have outliers in windspeed, humdity, casual
# outliers in casual doesn't effect the predictions as it is the target variable

outliers=c('humdity','windspeed')

#Replacing all outliers with NA

for (i in outliers) {
  val = b_rent[,i][b_rent[,i]%in% boxplot.stats(b_rent[,i])$out]
  print(length(val))
  b_rent[,i][b_rent[,i] %in% val] = NA
  
}
#Again checking for missing data after outliers
apply(b_rent, 2,function(x) {sum(is.na(x))})

b_rent= drop_na(b_rent)

#copying data
data_no_outliers =b_rent

                                        ######VISUALIZATION########
#PLOT BETWEEN TEMP AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$temp,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='temp',y='count')

#PLOT BETWEEN ATEMP AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$atemp,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='atemp',y='count')

#PLOT BETWEEN Humdity AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$humdity,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='humdtiy',y='count')

#PLOT BETWEEN Windspeed AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$windspeed,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='windspeed',y='count')

#PLOT BETWEEN season AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$season,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='season',y='count')


#PLOT BETWEEN MONTH AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$month,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='month',y='count')

#PLOT BETWEEN WEEKDAY AND COUNT

ggplot(data=b_rent, aes_string(x=b_rent$weekday,y=b_rent$count))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='weekday',y='count')



                                    ######FEATURE SELECTION #########

#Correlation plot Numerical data

corrgram(b_rent[,numeric_index], order = F, upper.panel = panel.pie,
         text.panel = panel.txt, main='Correlation Plot')


# check VIF

vif(b_rent[,10:15])

#As VIF is greater than 10 is not appropriate or multicollinerity

## Chi-square test

for (i in cat_names) {
  print(i)
  print(chisq.test(table(b_rent$count,b_rent[,i])))
  
}

#From correlation plot and VIF temp and atemp are highly correlated so removing atemp
#removing Casual and registered because their are targets
#from chi-squared test removing holiday because it does not contribute to the variables
#Removing date and instant not usefull in prediction

#removing unnecessary columns
b_rent = subset(b_rent, select=-c(date,instant,atemp,casual,registered,holiday))
View(b_rent)

#copying clean data
clean_data = b_rent

#Scaling categorical variable with dummies
install.packages("dummies") #for scaling
library(dummies)

#dummy.data.frame()
df_dum = dummy.data.frame(b_rent)
View(df_dum)

                                     #######MODELLING########

#Dividing data into to train and test
set.seed(270)
train_index = createDataPartition(df_dum$count,p=0.7,list = FALSE)
train = df_dum[train_index,]
test = df_dum[-train_index,]

###--DECISION_TREE--###

DT_model=rpart(count~ .,data = train,method = 'anova')

summary(DT_model)
#predict
pred_DT= predict(DT_model,test[,-34])

#Evaluation
regr.eval(test[,34],pred_DT)

#    mae          mse         rmse         mape 
#6.942517e+02 8.202756e+05 9.056907e+02 1.942659e-01 

# compute r^2
rss_dt = sum((pred_DT - test$count) ^ 2)
tss_dt = sum((test$count - mean(test$count)) ^ 2)
rsq_dt = 1 - rss_dt/tss_dt
rsq_dt

## r2 score 0.7645988


###---Random_Forest---###

RF_model=randomForest(count~ .,data = train)
#predict
pred_RF=predict(RF_model,test[,-34])
#evaluation
regr.eval(test[,34],pred_RF)

#mae          mse         rmse         mape 
#4.975377e+02 4.936155e+05 7.025777e+02 1.402441e-01 

##R2 score
rss_rf = sum((pred_RF - test$count) ^ 2)
tss_rf = sum((test$count - mean(test$count)) ^ 2)
rsq_rf = 1 - rss_rf/tss_rf
rsq_rf

## r2 score  0.8583431


####---Linear_Regression---####

lr_model = lm(count~ .,data = train)
summary(lr_model)

#predict
pred_LR =predict(lr_model,test[,-34])

regr.eval(test[,34],pred_LR)

# mae          mse         rmse         mape 
#5.867420e+02 6.886766e+05 8.298654e+02 1.720520e-01

#From summary
#R-squared:  0.861
