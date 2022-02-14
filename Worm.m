net=googlenet;
%Wczytanie danych
imds=imageDatastore("WormImages");
label_tab = readtable("WormData.csv");
%Podzia³ zbioru danych
imds.Labels = categorical(label_tab.Status);
[imdsTrain,imdsTest]=splitEachLabel(imds,.8,"randomized");
imdsTrain=augmentedImageDatastore([224 224],imdsTrain,"ColorPreprocessing","gray2rgb");
imdstest=augmentedImageDatastore([224 224],imdsTest,"ColorPreprocessing","gray2rgb");
%Ustawienia treningu i trening
train_options = trainingOptions("sgdm","InitialLearnRate", 0.001);
net = trainNetwork(imdsTrain,lgraph_2,train_options);
%Predykcja
preds=classify(net,imdstest);
%Ewaluacja danych
true_test=imdsTest.Labels;
numCorrect=nnz(true_test==preds)
fracCorrect=numCorrect/numel(preds)
confusionchart(imdsTest.Labels,preds)

