using TyImageProcessing
using TyBase
using TyDeepLearning
using TyMath
using TyPlot
using TyImages
using TySymbolicMath
set_backend(:mindspore)



layer_enc1=SequentialCell([
    convolution2dLayer(3, 64, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(64),
    reluLayer(),
    convolution2dLayer(64, 64, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(64),
    reluLayer(),
])

layer_enc2=SequentialCell([
    convolution2dLayer(64, 128, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(128),
    reluLayer(),
    convolution2dLayer(128, 128, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(128),
    reluLayer(),
])

layer_enc3=SequentialCell([
    convolution2dLayer(128, 256, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(256),
    reluLayer(),
    convolution2dLayer(256, 256, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(256),
    reluLayer(),
])

layer_bridge=SequentialCell([
    convolution2dLayer(256, 512, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(512),
    reluLayer(),
    convolution2dLayer(512, 512, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(512),
    reluLayer(),
])

layer_dec1=SequentialCell([
    convolution2dLayer(256, 256, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(256),
    reluLayer(),
    convolution2dLayer(256, 256, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(256),
    reluLayer(),
])

layer_dec2=SequentialCell([
    convolution2dLayer(128, 128, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(128),
    reluLayer(),
    convolution2dLayer(128, 128, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(128),
    reluLayer(),
])

layer_dec3=SequentialCell([
    convolution2dLayer(64, 64, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(64),
    reluLayer(),
    convolution2dLayer(64, 64, 3; Stride = 1, PaddingMode="pad",PaddingSize=1),
    batchNormalization2dLayer(64),
    reluLayer(),
])


layer_res1 = SequentialCell([
    ("pool3", maxPooling2dLayer(2; Stride=2)),
    ("bridge", layer_bridge),
    ("up1", transposedConv2dLayer(512, 256, 2; Stride = 2)),
        
])

layer_res1 = SequentialCell([
    ("pool3", maxPooling2dLayer(2; Stride=2)),
    ("bridge", layer_bridge),
    ("up1", transposedConv2dLayer(512, 256, 2; Stride = 2)),
        
])

layer_res2 = SequentialCell([
    ("pool2", maxPooling2dLayer(2; Stride=2)),
    ("res", Residual_Block(layer_enc3, layer_res1, reluLayer())), 
    ("dec1", layer_dec1), 
    ("up2", transposedConv2dLayer(256, 128, 2; Stride = 2)),      
        
])

layer_res3 = SequentialCell([
    ("pool1", maxPooling2dLayer(2; Stride=2)),
    ("res2", Residual_Block(layer_enc2, layer_res2, reluLayer())), 
    ("dec2", layer_dec2), 
    ("up3", transposedConv2dLayer(128, 64, 2; Stride = 2)),       
        
])


layer = SequentialCell([
    ("res3", Residual_Block(layer_enc1, layer_res3, reluLayer())),
    ("dec3", layer_dec3), 
    ("out", convolution2dLayer(64, 3, 1;)), 
    ("act", sigmoidLayer()), 
    

])


lgraph = layerGraph(layer)
lgplot(lgraph)