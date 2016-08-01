dataDirectory <- "H:/R/"
data <- read.csv(paste(dataDirectory, 'sunnyData.csv', sep=""), header = TRUE)
#data
dtMatrix <- create_matrix(data["Text"])
container <- create_container(dtMatrix, data$IsSunny, trainSize=1:45, virgin=FALSE)

# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)
predictionData <- list("sunny sunny sunny rainy rainy", "rainy sunny rainy rainy", "mist sunny","fog rainy", "rainy rainy")
trace("create_matrix", edit=T)
predMatrix <- create_matrix(predictionData, originalMatrix=dtMatrix)
predSize = length(predictionData);
predictionContainer <- create_container(predMatrix, labels=rep(0,predSize), testSize=1:predSize, virgin=FALSE)
results <- classify_model(predictionContainer, model)
results

