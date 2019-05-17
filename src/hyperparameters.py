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
encPertMean = 0
encPertStd = .3
encPertAlpha = 1
encPertBeta = 1
encPertGamma = 1

archPercLossAlpha = 1
archPercLossBeta = 1.3
archLossPowAlpha = 1.5
archLossPowBeta = 1.5
archL1L2LossAlpha = 1
archL1L2LossBeta = 1
archL1L2PertMean = 0
archL1L2PertStd = .1
archL1L2PertAlpha = 1
archL1L2PertBeta = 1
archL1L2PertGamma = .1
archLossAlpha = 1.5
archSubsetLossBeta = 1
archSubsetLossGamma = 1
archMainLossPow = 1.5
archSubsetLossPow = 1

gloEpochsNum = 500
gloBatchSize = 35
gloEvalEvery = 20

encEpochsNum = 500
encBatchSize = 35
encEvalEvery = 5

archEpochsNum = 151
archSubsetBatchSize = 35
archMainBatchSize = 75
archEvalEvery = 10
archRatio = (0, 1)
