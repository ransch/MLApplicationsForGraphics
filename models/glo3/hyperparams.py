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
glo2MainCoeff = 1

gloPertMean = 0  # Noise Sampling
gloPertStd = 1  # Noise Sampling
gloPertCoeff = .5  # Noise term multiplier

gloEpochsNum = 501
gloBatchSize = 35
gloEvalEvery = 100

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

encEpochsNum = 1501
encBatchSize = 35
encEvalEvery = 20

###################################################################
#                             Arch Config
###################################################################

archGenAdamLr = .001
archGenAdamBetas = (0.9, 0.999)
archEncAdamLr = .001
archEncAdamBetas = (0.9, 0.999)

archPercLossAlpha = 1  # Perceptual loss  multiplier l1 norm
archLossPowAlpha = 1  # Perceptual loss  pow l1 norm

archPercLossBeta = 1  # Perceptual loss  multiplier features
archLossPowBeta = 1  # Perceptual loss  pow features

archL1L2LossAlpha = 1  # loss1[Zx, E(x)] l1norm
archL1L2LossBeta = 1  # loss1[Zx, E(x)] l2norm

archPertMean = 0  # Noise Sampling
archPertStd = 0  # Noise Sampling
archPertCoeff = 0  # Noise term multiplier
archPertPow = 0  # Noise term pow

archLossAlpha = 1.3  # Loss3 coeff (main) - without noise
archSubsetLossBeta = .3  # Loss1 coeff (subset)
archSubsetLossGamma = 1.3  # Loss2 coeff (subset)

archMainLossPow = 1.3  # Loss3 pow (main) - without noise
archSubsetLossPow = 1.3  # Loss1 and Loss2 pow (subset)

archEpochsNum = 11
archSubsetBatchSize = 50
archMainBatchSize = 150
archEvalEvery = 2
archRatio = (0, 1)

###################################################################
#                          Clustering Config
###################################################################

clusteringPCADim = 100
clusteringNum = 128
clusteringReprNum = 8
