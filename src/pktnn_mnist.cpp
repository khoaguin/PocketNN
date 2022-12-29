#include "pktnn_mnist.h"

int fc_int_dfa_mnist() {
    // Loading the MNIST dataset
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktnn::pktmat mnistTrainLabels;
    pktnn::pktmat mnistTrainImages;
    pktnn::pktmat mnistTestLabels;
    pktnn::pktmat mnistTestImages;

    pktnn::pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktnn::pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1

    std::cout << "Loaded train images: " << mnistTrainImages.rows() << ".\nLoaded test images: " << mnistTestImages.rows() << "\n";

    // mnistTestImages.printMat(std::cout);

    // Defining the network
    int numClasses = 10;
    int mnistRows = 28;
    int mnistCols = 28;

    const int dimInput = mnistRows * mnistCols;
    const int dim1 = 100;
    const int dim2 = 50;
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_tanh;

    pktnn::pktfc fc1(dimInput, dim1);
    pktnn::pktfc fc2(dim1, dim2);
    pktnn::pktfc fcLast(dim2, numClasses);
    fc1.useDfa(true).setActv(a).setNextLayer(fc2);
    fc2.useDfa(true).setActv(a).setNextLayer(fcLast);
    fcLast.useDfa(true).setActv(a);

    // initialization
    pktnn::pktmat trainTargetMat(numTrainSamples, numClasses);
    pktnn::pktmat testTargetMat(numTestSamples, numClasses);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    for (int r = 0; r < numTrainSamples; ++r) {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect: " << numCorrect << "\n";
    fc1.printWeight(std::cout);

    return 0;
};