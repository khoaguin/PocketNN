#include "pocketnn.h"

int pocketnn::fc_int_dfa_mnist() {
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktnn::pktmat mnistTrainLabels;
    pktnn::pktmat mnistTrainImages;
    pktnn::pktmat mnistTestLabels;
    pktnn::pktmat mnistTestImages;

    // pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktnn::pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    // pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    // pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)

    
    std::cout << "Loading train images: " << numTrainSamples << ". Loaded test images: " << numTestSamples << "\n";


    return 0;
};