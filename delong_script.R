# Created 5-2-2020
# Calculate DeLong test for AUC curve comparison

library(pROC)

response<-c(0,0,1,1,1)
modela<-c(0.1,0.2,0.6,0.7,0.8)
modelb<-c(0.3,0.6,0.2,0.7,0.9)
roca <- roc(response,modela)
rocb<-roc(response,modelb)

roc.test(roca, rocb, method=c("delong"))