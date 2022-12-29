#include <iostream>

#include <pocketnn/pocketnn.h>

// using namespace pktnn;


int main() {
    int numTrainSamples = 60000;
    int numTestSamples = 10000;
    std::cout << "We are in main!!!" << std::endl;
    pocketnn::fc_int_dfa_mnist();

    return 0;
}