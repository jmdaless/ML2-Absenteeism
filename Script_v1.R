
########################  PACKAGES ########################  
knitr::opts_chunk$set(echo = TRUE)
library(ggplot2)
library(e1071)     # Alternative for Skewness
library(glmnet)    # Lasso
library(caret)     # To enable Lasso training with CV.
#library(FSelector) # To compute Information Gain
library(dplyr)
library(stats)
library(purrr)
library(tidyr)
library(questionr)


set.seed(123)
path<- "/Users/estevezmelgarejoramon/Desktop/ML2/FirstAssignment/Absenteeism"
setwd(path)

rm(list = ls())
########################  FUNCTIONS ########################  
lm.model <- function(training_dataset) {
  set.seed(123)
  # Create a training control configuration that applies a 5-fold cross validation
  train_control_config <- trainControl(method = "repeatedcv", 
                                       number = 5, 
                                       repeats = 1,
                                       returnResamp = "all")
  
  # Fit a glm model to the input training data
  this.model <- train(Absenteeism ~ .,
                      data = training_dataset, #the dataset is what will change 
                      method = "glm", 
                      metric = "Accuracy",
                      preProc = c("center", "scale"),
                      trControl=train_control_config)
  
  return(this.model)
}




########################  IMPORT DATA ######################
training = read.csv(file = file.path("Absenteeism_at_work_classification_training.csv"),  header = TRUE, dec = ".", sep = ";")
test = read.csv(file = file.path("Absenteeism_at_work_classification_test.csv"),  header = TRUE, dec = ".", sep = ";")

############################  EDA  ##########################

# We first join train and test sets so that we can treat them equaly
test$Absenteeism <- 0
dataset <- rbind(training, test)
dataset <- dataset[,!(names(dataset) %in% c("ID"))]

# Renaming columns so that they fit better in plots!
names(dataset) <- recode(names(dataset),
                         "Transportation.expense" = "TranspExp",
                         "Hit.target"  ="Hit",
                         "Social.smoker"  = "Smoker",
                         "Distance.from.Residence.to.Work" = "WorkDist",
                         "Disciplinary.failure" = "DiscFailure",
                         "Month.of.absence" = "MonthOfAbs",
                         "ID.Worker" = "WorkerID",
                         "Service.time" = "ServTime",
                         "Day.of.the.week"   = "WeekDay",
                         "Work.load.Average.day"  = "WorkLoadAvgDay",
                         "Social.drinker"  = "Drinker",
                         "Body.mass.index" = "BodyMassIndex",
                         "Reason.for.absence"  = "Reason")



# Now we pint summary
summary(dataset)

# Check for NA´s, we can see that there are none!
colSums(is.na(dataset))

# factor will convert the column to categorical. lapply applies the factor function to each of the columns in categorical_columns 
categorical_columns <- c('WorkerID','Reason', 'MonthOfAbs', 'WeekDay', 'Seasons', 'DiscFailure', 'Drinker', 'Smoker', 'Absenteeism')
dataset[categorical_columns] <- lapply(dataset[categorical_columns], factor) 

# Checking that type was changed
summary(dataset)
str(dataset)



# Recoding variables for better understanding
dataset$Reason <- recode(dataset$Reason, 
                         '0'='InfParasDis', '1'='Neoplasms', '2'='DisOfBblood', '3'='Endoc&metDis', '4'='MentAndBehavDisor',
                         '5'='NervSys', '6'='EyeAndBdnexa', '7'='EarAndMast', '8'='CircSys', '9'='RespSys', '10'='DigSys', 
                         '11'='SkinAndSaubcutTiss', '12'='MuscuAndConnect',  '13'='Genitourinary', '14'='PregnanBirthAndPuerp', 
                         '15'='Perinatal', '16'='Congenital', '17'='AbnormalFindings', '18'='InjuryPoisoningAndOther ',
                         '19'='MorbidityAndMortality', '21'='!influencingHealth', '22'='patientFollowUp',
                         '23'='Consultation', '24'='BloodDonation', '25'='LabExamination', '26'='Unjustified',
                         '27'='Physiotherapy', '28'='Dental')

dataset$Seasons =recode(dataset$Seasons,'1'='summer','2'='autumn','3'='winter','4'='spring')
dataset$Education =recode(dataset$Education, '1'='highschool','2'='graduate','3'='postgraduate','4'='master&PhD')

# Base line model


suppressWarnings({baseline_model <- lm.model(dataset[1:594,])})# Only the training data 
confusionMatrix(baseline_model, "none")
plot(varImp(baseline_model, scale = T), top = 20)


# Histogram of numeric variables (bins = 30)
dataset[1:594,] %>% 
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()


# bucketize some features:
dataset$Son <-as.factor(ifelse(dataset$Son == 0, "No", "Yes"))
dataset$Pet <-as.factor(ifelse(dataset$Pet == 0, "No", "Yes"))


# body mass index
# Deberiamos hacerlos factores??????????
dataset$BodyMassIndex <-.bincode(dataset$BodyMassIndex, c(18.5, 24.9, 29.99, 34.9, 39.99), TRUE, TRUE)
dataset$BodyMassIndex <- as.factor(recode(dataset$BodyMassIndex, '1' = "Normal", '2' = "Overweight", '3' = "Obesity", '4' = "Extreme_Obesity"))

# Age
dataset$Age <-as.factor(.bincode(dataset$Age, c(18, 30, 45, 60), TRUE, TRUE))



# GROUP CATEGORIES. count the number of categories on each cat variable. if there are many with low occurence we might group them.
sapply(dataset %>% keep(is.factor), function(x){length(unique(x))})  ### Too many reasons for absence!

# reason for absence frequencies
ReasonFrec <- freq(dataset$Reason, digits = 5, valid = F)
ReasonFrec <- tibble::rownames_to_column(ReasonFrec, "Reason") # La funcion frec() deja Reasons como rownames, asi los hago una columna para el ggplot


ggplot(ReasonFrec , aes(x = reorder(Reason, - `%`), y = `%`)) +    
  geom_col() +
  ylab("%") +
  xlab("Reason") +
  coord_flip()

# Let's remove airports with less or equal than 1% frequency
reasonsToGroup <- ReasonFrec[ReasonFrec$`%` <= 1, "Reason"]

dataset$Reason <- as.character(dataset$Reason) # We have to convert to character and then back
dataset[dataset$Reason %in% reasonsToGroup, "Reason"] <- "OTHERS"
dataset$Reason <- as.factor(dataset$Reason)

freq(dataset$Reason, digits = 5, valid = F)


# Remove employee.id as may mislead the analysis for future employees.
dataset <- dataset[,!(names(dataset) %in% c("WorkerID"))]

#Log of distance to work
dataset$log_dist <-log(dataset$WorkDist)

# Box plot of numeric variables
numVars<- dplyr::select_if(dataset[1:594,], is.numeric)
ggplot(stack(numTrain), aes(x = ind, y = values)) +
      stat_boxplot(coef = 3) + 
      xlab("") +
      theme(axis.text.x =  element_text(angle = 90))



# Creo que esto se puede hacer sin for loop, lo investigare! tambien No se si alguno de los outlier estaban en el test set, no deberiamos quitarlos en ese caso no?
for (col in names(dataset)) {
  if (is.numeric(dataset[[col]]) && col != "Absenteeism"){
    print(ggplot(dataset, aes_string(y=col))+ geom_boxplot(width=0.1) + theme(axis.line.x=element_blank(),axis.title.x=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(),legend.position="none"))
    to_remove <- boxplot.stats(dataset[[col]], coef = 3)$out
    dataset_no_outliers <- dataset[!dataset[[col]] %in% to_remove, ]
  }
}




# Still with outliers..
training <- dataset[1:594,]
test <- dataset[595:740,]


# Juan me planto aquí....
# No consigo instalar la maldita biblioteca FSelector y no me da la vida para buscar alternativas,
# Creo que he mejorado un poco el código hasta aqui, mañana seguire dandole duro.
# Hablamos!










