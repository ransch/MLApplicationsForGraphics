latentDim = 100  # The dimension of the latent space
genFeatureMapsSize = 64  # The size of the feature maps in the generator
encFeatureMapsSize = 64  # The size of the feature maps in the decoder

###################################################################
#                             Glo Config
###################################################################

genAdamLr = .001
genAdamBetas = (0.9, 0.999)
embedAdamLr = .001
embedAdamBetas = (0.9, 0.999)

gloEvalAdamLr = .001
gloEvalAdamBetas = (0.9, 0.999)

gloLossAlpha = 1
gloLossBeta = 1
gloLossPowAlpha = 1
gloLossPowBeta = 1

glo2SubsetCoeff = 1
glo2MainCoeff = 2

gloPertMean = 0  # Noise Sampling
gloPertStd = .4  # Noise Sampling
gloPertCoeff = .7  # Noise term multiplier

gloEpochsNum = 4001
gloBatchSize = 75
gloEvalEvery = 250

glo2SubsetBatchSize = 35
glo2MainBatchSize = 50

gloEvalEpochsNum = 10 ** 5

###################################################################
#                           Encoder Config
###################################################################

encAdamLr = .001
encAdamBetas = (0.9, 0.999)

encLossAlpha = 1
encLossBeta = 1

encEpochsNum = 5001
encBatchSize = 75
encEvalEvery = 20

###################################################################
#                          Clustering Config
###################################################################

clusteringPCADim = 100
clusteringNum = 128
clusteringReprNum = 8
