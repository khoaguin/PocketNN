#include "pktnn_examples.h"

int fc_int_bp_simple() {
    // constructing the neural net
    const int dim1 = 3;
    const int dim2 = 5;
    const int numEpochs = 5;

    pktnn::pktmat mat1(1, dim1);
    pktnn::pktfc fc1(dim1, dim2);
    pktnn::pktfc fc2(dim2, 1);

    std::cout << "--- Weights before initialization --- \n";
    fc1.printWeight(std::cout);
    fc2.printWeight(std::cout);

    // initialize the weights and biases
    fc1.useDfa(false).initHeWeightBias().setActv(pktnn::pktactv::Actv::pocket_tanh).setNextLayer(fc2);
    fc2.useDfa(false).initHeWeightBias().setActv(pktnn::pktactv::Actv::as_is).setPrevLayer(fc1);
    
    std::cout << "--- Weights after initialization --- \n";
    fc1.printWeight(std::cout);
    fc2.printWeight(std::cout);

    // dummy data
    mat1.setElem(0, 0, 10);
    mat1.setElem(0, 1, 20);
    mat1.setElem(0, 2, 30);
    std::cout << "--- Data --- \n";
    mat1.printMat(std::cout);

    int y = 551; // random number

    for (int i = 0; i < numEpochs; ++i) {
        fc1.forward(mat1);
        fc2.forward(fc1);

        int y_hat = fc2.mOutput.getElem(0, 0);
        int loss = pktnn::pktloss::scalarL2Loss(y, y_hat);
        int lossDelta = pktnn::pktloss::scalarL2LossDelta(y, y_hat);
        std::cout << "y: " << y << ", y_hat: " << y_hat << ", l2 loss: " << loss << ", l2 loss delta: " << lossDelta << "\n";

        pktnn::pktmat lossDeltaMat;
        lossDeltaMat.resetZero(1, 1).setElem(0, 0, lossDelta);

        fc2.backward(lossDeltaMat, 1e5);
    }

    return 0;
};

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
    // mnistTestImages.printMat(std::cout);
    std::cout << "Loaded train images: " << mnistTrainImages.rows() << ".\nLoaded test images: " << mnistTestImages.rows() << "\n";

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
    fcLast.printWeight(std::cout);
    // initialization
    // pktnn::pktmat trainTargetMat(numTrainSamples, numClasses);
    // pktnn::pktmat testTargetMat(numTestSamples, numClasses);

    // int numCorrect = 0;
    // fc1.forward(mnistTrainImages);
    // for (int r = 0; r < numTrainSamples; ++r) {
    //     trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
    //     if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
    //         ++numCorrect;
    //     }
    // }
    // std::cout << "Initial training numCorrect: " << numCorrect << "\n";

    return 0;
};