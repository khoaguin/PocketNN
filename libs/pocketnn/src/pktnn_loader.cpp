#include "pktnn_loader.h"

using namespace pktnn;

void pktnn::pktloader::csvLoader(pktmat& saveToMat, std::string fileName) {
    std::ifstream dataFile(fileName);
    if (dataFile.fail()) {
        std::cout << fileName << " does not exist!\n";
        return;
    }

    std::string oneLine;
    std::getline(dataFile, oneLine); // read the header and discard
    std::stringstream check1(oneLine);
    std::string tokenized;
    char token = ',';
    int numCols = 0;
    int numRows = 0;

    while (std::getline(check1, tokenized, token)) {
        ++numCols;
    }

    while (std::getline(dataFile, oneLine)) {
        ++numRows;
    }

    std::cout << "Rows, Cols: " << numRows << ", " << numCols << "\n";

    check1.clear();
    check1.str(std::string());

    dataFile.clear();
    dataFile.seekg(0);
    std::getline(dataFile, oneLine); // read the header and discard
    std::cout << oneLine << "\n";
    saveToMat.resetZero(numRows, numCols);

    int r = 0;
    while (std::getline(dataFile, oneLine)) {
        int c = 0;
        check1.clear();
        check1.str(oneLine);
        while (std::getline(check1, tokenized, token)) {
            saveToMat.setElem(r, c, std::stoi(tokenized)); // round off
            ++c;
        }
        ++r;
    }

    dataFile.close();
}

// bool pktloader::file_exists(const std::string& name) {
//     FILE* pFile = NULL;
//     if (fopen_s(&pFile, name.c_str(), "r") == 0) {
//         fclose(pFile);
//         return true;
//     }
//     else {
//         return false;
//     }
// }

// void pktnn::_downloadDataset(pktloader::Dataset dataset) {
//     std::string fileName = "dataset/diabetes.tab.txt";
//     if (pktloader::file_exists(fileName)) {
//         std::cout << "File already exists: " << fileName << "\n";
//     } else {
//         std::string diabetesUrl = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";
//         std::wstring wideDiabetesUrl = std::wstring(diabetesUrl.begin(), diabetesUrl.end());
//         std::wstring wideFileName = std::wstring(fileName.begin(), fileName.end());
//         URLDownloadToFile(NULL, wideDiabetesUrl.c_str(), wideFileName.c_str(), BINDF_GETNEWESTVERSION, NULL);
//     }
// }

// void pktloader::downloadDataset(pktloader::Dataset dataset) {
//     _downloadDataset(dataset);
// }

void pktnn::pktloader::parseDatasetDiabetes(pktmat& saveToMat, std::string fileName) {
    std::ifstream dataFile(fileName); //fstream
    if (dataFile.fail()) {
        // TODO: error handling
    }

    std::string oneLine;
    std::getline(dataFile, oneLine); // read the header and discard
    std::stringstream check1(oneLine);
    std::string tokenized;
    int numCols = 0;
    int numRows = 0;

    while (std::getline(check1, tokenized, '\t')) {
        ++numCols;
    }

    while (std::getline(dataFile, oneLine)) {
        ++numRows;
    }

    check1.clear();
    check1.str(std::string());

    dataFile.clear();
    dataFile.seekg(0);
    std::getline(dataFile, oneLine); // read the header and discard
    saveToMat.resetZero(numRows, numCols);

    int r = 0;
    while (std::getline(dataFile, oneLine)) {
        int c = 0;
        check1.clear();
        check1.str(oneLine);        
        while (std::getline(check1, tokenized, '\t')) {
            saveToMat.setElem(r, c, std::stoi(tokenized)); // round off
            ++c;
        }
        ++r;
    }

    dataFile.close();
}

int pktnn::pktloader::reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

pktmat3d** pktnn::pktloader::loadMnistImages(int numImagesToLoad) {
    std::ifstream file("data/mnist/train-images.idx3-ubyte", std::ios::binary);
    pktmat3d** imageBatches = nullptr;
    if (file.is_open()) {
        // format for the beginning of the file (32 bits each)
        // should be 2051, 60000, 28, 28
        int magicNumber = 0;
        int numItems = 0;
        int numRows = 0;
        int numCols = 0;
        // assumed to be assigned in advance
        imageBatches = new pktmat3d* [numImagesToLoad];

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = pktloader::reverseInt(magicNumber);

        file.read((char*)&numItems, sizeof(numItems));
        numItems = pktloader::reverseInt(numItems);

        file.read((char*)&numRows, sizeof(numRows));
        numRows = pktloader::reverseInt(numRows);

        file.read((char*)&numCols, sizeof(numCols));
        numCols = pktloader::reverseInt(numCols);

        std::cout << magicNumber << ", " << numItems << ", " << numRows << ", " << numCols << "\n";

        for (int i = 0; i < numImagesToLoad; ++i) {
            imageBatches[i] = new pktmat3d(1, numRows, numCols);
            for (int r = 0; r < numRows; ++r) {
                for (int c = 0; c < numCols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    imageBatches[i]->setElem(0, r, c, temp);
                }
            }
        }
    }
    else {
        std::cout << "Dataset is not loaded properly.\n";
    }
    return imageBatches;
}

void pktnn::pktloader::loadMnistImages(pktmat& images, int numImagesToLoad, bool isTrain) {
    std::string fileName;
    if (isTrain) {
        fileName = "data/mnist/train-images.idx3-ubyte";
    }
    else {
        fileName = "data/mnist/t10k-images.idx3-ubyte";
    }

    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open()) {
        // format for the beginning of the file (32 bits each)
        // should be 2051, 60000, 28, 28
        int magicNumber = 0;
        int numItems = 0;
        int numRows = 0;
        int numCols = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        file.read((char*)&numItems, sizeof(numItems));
        file.read((char*)&numRows, sizeof(numRows));
        file.read((char*)&numCols, sizeof(numCols));

        magicNumber = pktloader::reverseInt(magicNumber);
        numItems = pktloader::reverseInt(numItems);
        numRows = pktloader::reverseInt(numRows);
        numCols = pktloader::reverseInt(numCols);

        std::cout << "Loading " << fileName << "\n";
        std::cout << "Magic number: " << magicNumber
                  << ", Total items: " << numItems
                  << ", Rows: " << numRows
                  << ", Cols: " << numCols << "\n";

        images.resetZero(numImagesToLoad, numRows * numCols);
        for (int i = 0; i < numImagesToLoad; ++i) {
            for (int r = 0; r < numRows; ++r) {
                for (int c = 0; c < numCols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    int value = temp;
                    images.setElem(i, r * numCols + c, value);
                }
            }
        }
    }
    else {
        std::cout << "Dataset is not loaded properly.\n";
    }
}

void pktnn::pktloader::loadMnistLabels(pktmat& labels, int numLabelsToLoad, bool isTrain) {
    std::string fileName;
    if (isTrain) {
        fileName = "data/mnist/train-labels.idx1-ubyte";
    }
    else {
        fileName = "data/mnist/t10k-labels.idx1-ubyte";
    }

    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open()) {
        // format for the beginning of the file (32 bits each)
        // should be 2049, 60000
        int magicNumber = 0;
        int numItems = 0;
        labels.resetZero(numLabelsToLoad, 1);

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = pktloader::reverseInt(magicNumber);

        file.read((char*)&numItems, sizeof(numItems));
        numItems = pktloader::reverseInt(numItems);

        std::cout << "Loading " << fileName << "\n";
        std::cout << "Magic number: " << magicNumber << ", Total items: " << numItems << "\n";
        
        for (int i = 0; i < numLabelsToLoad; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels.setElem(i, 0, temp);
        }
    }
    else {
        std::cout << "Failed to load the label dataset.\n";
    }
}


void pktnn::pktloader::loadFashionMnistImages(pktmat& images, int numImagesToLoad, bool isTrain) {
    std::string fileName;
    if (isTrain) {
        fileName = "data/fashion_mnist/train-images-idx3-ubyte";
    }
    else {
        fileName = "data/fashion_mnist/t10k-images-idx3-ubyte";
    }

    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open()) {
        // format for the beginning of the file (32 bits each)
        // should be 2051, 60000, 28, 28
        int magicNumber = 0;
        int numItems = 0;
        int numRows = 0;
        int numCols = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        file.read((char*)&numItems, sizeof(numItems));
        file.read((char*)&numRows, sizeof(numRows));
        file.read((char*)&numCols, sizeof(numCols));

        magicNumber = pktloader::reverseInt(magicNumber);
        numItems = pktloader::reverseInt(numItems);
        numRows = pktloader::reverseInt(numRows);
        numCols = pktloader::reverseInt(numCols);

        std::cout << "Loading " << fileName << "\n";
        std::cout << "Magic number: " << magicNumber
            << ", Total items: " << numItems
            << ", Rows: " << numRows
            << ", Cols: " << numCols << "\n";

        images.resetZero(numImagesToLoad, numRows * numCols);
        for (int i = 0; i < numImagesToLoad; ++i) {
            for (int r = 0; r < numRows; ++r) {
                for (int c = 0; c < numCols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    int value = temp;
                    images.setElem(i, r * numCols + c, value);
                }
            }
        }
    }
    else {
        std::cout << "Dataset is not loaded properly.\n";
    }
}

void pktnn::pktloader::loadFashionMnistLabels(pktmat& labels, int numLabelsToLoad, bool isTrain) {
    std::string fileName;
    if (isTrain) {
        fileName = "data/fashion_mnist/train-labels-idx1-ubyte";
    }
    else {
        fileName = "data/fashion_mnist/t10k-labels-idx1-ubyte";
    }

    std::ifstream file(fileName, std::ios::binary);
    if (file.is_open()) {
        // format for the beginning of the file (32 bits each)
        // should be 2049, 60000
        int magicNumber = 0;
        int numItems = 0;
        labels.resetZero(numLabelsToLoad, 1);

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = pktloader::reverseInt(magicNumber);

        file.read((char*)&numItems, sizeof(numItems));
        numItems = pktloader::reverseInt(numItems);

        std::cout << "Loading " << fileName << "\n";
        std::cout << "Magic number: " << magicNumber << ", Total items: " << numItems << "\n";

        for (int i = 0; i < numLabelsToLoad; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels.setElem(i, 0, temp);
        }
    }
    else {
        std::cout << "Failed to load the label dataset.\n";
    }
}
