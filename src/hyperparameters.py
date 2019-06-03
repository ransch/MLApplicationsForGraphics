latentDim = 100  # The dimension of the latent space

genFeatureMapsSize = 64  # The size of the feature maps in the generator

encFeatureMapsSize = 64  # The size of the feature maps in the decoder

genAdamLr = .001
genAdamBetas = (0.9, 0.999)
embedAdamLr = .001
embedAdamBetas = (0.9, 0.999)

encAdamLr = .001
encAdamBetas = (0.9, 0.999)

archGenAdamLr = .001
archGenAdamBetas = (0.9, 0.999)
archEncAdamLr = .001
archEncAdamBetas = (0.9, 0.999)

gloLossAlpha = 1
gloLossBeta = 1
gloLossPowAlpha = 1
gloLossPowBeta = 1

encLossAlpha = 1
encLossBeta = 1

###################################################################
#                             Arch Config
###################################################################
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
###################################################################

gloEpochsNum = 1001
gloBatchSize = 35
gloEvalEvery = 20

encEpochsNum = 501
encBatchSize = 35
encEvalEvery = 20

archEpochsNum = 11
archSubsetBatchSize = 50
archMainBatchSize = 150
archEvalEvery = 2
archRatio = (0, 1)

clusteringPCADim = 100
clusteringNum = 128
clusteringReprNum = 8
