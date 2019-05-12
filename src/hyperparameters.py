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

archPercLossAlpha = 1.5
archPercLossBeta = 1
archLossPowAlpha = 1.5
archLossPowBeta = 1
archL1L2LossAlpha = 1
archL1L2LossBeta = 1
archSubsetLossAlpha = 2
archSubsetLossBeta = 1
archSubsetLossGamma = 2
archMainLossPow = 3
archSubsetLossPow = 3

# gloEpochsNum = 500
# gloBatchSize = 35
# gloEvalEvery = 20
gloEpochsNum = 50
gloBatchSize = 75
gloEvalEvery = 10

encEpochsNum = 500
encBatchSize = 35
encEvalEvery = 2

archEpochsNum = 30
archSubsetBatchSize = 35
archMainBatchSize = 75
archEvalEvery = 2
archRatio = (3, 3)
