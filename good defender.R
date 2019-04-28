install.packages("dplyr")
library(dplyr)
library(ggplot2)
fifaData <- read.csv("C://Users//Damanpreet//Desktop//DV_CA2//CA2//CompleteDataset.csv")
#Negating for Goal Keepers
fifaData <- mutate(fifaData,Preferred.Positions = as.character(Preferred.Positions))
fifaData <- filter(fifaData,Preferred.Positions != "GK ")
fifaData<- select(fifaData,-GK.diving,-GK.handling,-GK.kicking,-GK.positioning,-GK.reflexes)
#Player attributes and defence scores
attributesData <- select(fifaData,c(14:42))
attributesData <- sapply( attributesData, as.numeric )
defencePositionData <- select(fifaData,CB,LB,LCB,LWB,RB,RCB,RWB)
#Correlating Player attributes and defence scores
defencePositionData <- select(fifaData,CB,LB,LCB,LWB,RB,RCB,RWB)
defenceCorr<- cor(attributesData,defencePositionData)
defenceCorr <- as.data.frame(defenceCorr)
head(defenceCorr)
#Clustering player attributes
defenceDist <- dist(defenceCorr, method = "euclidean")
defenceFit <- hclust(defenceDist, method="ward")
plot(defenceFit)
groups <- cutree(defenceFit, k=4)
#understanding clusters
defenceCorr <- mutate(defenceCorr, Attributes = row.names(defenceCorr), Cluster = groups)
defenceCorr
#NamingClusters
Advance <- filter(defenceCorr,Cluster == 1)$Attributes #Advance
BasicDefence <- filter(defenceCorr,Cluster == 2)$Attributes #Basic Defence
AttackSupport <- filter(defenceCorr,Cluster == 3)$Attributes # Attack Support
SupplemetalDefence <- filter(defenceCorr,Cluster == 4)$Attributes # Supplemetal Defence
Advance
#Averaging Co-relation
assignDefenceNames <- function(ClusterGroups)
{
  switch(ClusterGroups, "Advance", "BasicDefence", "AttackSupport", "SupplementalDefence")
}
defenceCorr <- defenceCorr%>%   rowwise() %>%   mutate(ClusterName = assignDefenceNames(Cluster))



defenceInsights <- as.data.frame(summarize(group_by(defenceCorr,ClusterName),CBMean = mean(CB), 
                                           RCBMean = mean(RCB),LCBMean = mean(LCB), RBMean = mean(RB),
                                           LBMean = mean(LB),RWBMean = mean(RWB), LWBMean = mean(LWB)))
select(defenceInsights, ClusterName, CBMean, RBMean, RWBMean)
#What makes a Good Center Back
ggplot(data=defenceInsights, aes(x=ClusterName, y=CBMean)) +
  geom_bar(stat="identity")+coord_flip() + ggtitle("What Makes a Good Center Back") +
  theme(axis.title.x=element_blank(),axis.title.y=element_blank() )
#What Makes a Good Full Back
ggplot(data=defenceInsights, aes(x=ClusterName, y=RBMean)) +
  geom_bar(stat="identity")+coord_flip()+ ggtitle("What Makes a Good Full Back") +
  theme(axis.title.x=element_blank(),axis.title.y=element_blank() )
#What Makes a Good Wing Back
ggplot(data=defenceInsights, aes(x=ClusterName, y=RWBMean)) +
  geom_bar(stat="identity")+coord_flip()+ggtitle("What Makes a Good Wing Back") +
  theme(axis.title.x=element_blank(),axis.title.y=element_blank() )
