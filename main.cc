#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>


struct CostFunction {
    double (*fn)(double a, double y);
    double (*delta)(double z, double a, double y);
};

typedef struct CostFunction CostFunction;

double sigmoid(double x)
{
    return 1.0/(1.0 + exp(-x));
}
double sigmoidPrime(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

double quadraticCostFn(double a, double y)
{
    return 0.5 * (a - y) * (a - y);
} 

double quadraticCostDelta(double z, double a, double y)
{
    return (a - y) * sigmoidPrime(z);
} 

double xEntropyCostFn(double a, double y)
{
    return -y*log(a) - (1-y)*log(1-a);
} 

double xEntropyCostDelta(double z, double a, double y)
{
    return (a - y);
} 


void printImage(double* input)
{
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%d", (int)(input[i*28+j] > 0.5));
        }
        printf("\n");
    } 
    printf("You should read %f\n", input[28*28]);
}

void feedForward(const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights, double* input, double** output)
{
    double* xPrev = NULL; 
    double* xNext = NULL;
    for (int i = 0; i < nLayer-1; i++) {

        if (i==0) {
            xPrev = input;
            //printImage(input);
            //printImage(xPrev);
        }
        else {
            xPrev = xNext;
        }

        xNext = (double*) calloc(sizes[i+1], sizeof(xNext)); 

        for (int j = 0; j < sizes[i+1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                xNext[j] += weights[i][j][k] * xPrev[k];
            }
            xNext[j] += biases[i][j];
            xNext[j] = sigmoid(xNext[j]);
            //printf("xNext[%d] = %f\n", j, xNext[j]);
        }


        if (i!=0) {
            free(xPrev);
            xPrev = NULL;
        }
    }
    if ((*output) != NULL) {
        free((*output));
        (*output) = NULL;
    }
    (*output) = xNext;
}

void shuffle(unsigned int* v, const unsigned int n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            unsigned int t = v[j];
            v[j] = v[i];
            v[i] = t;
        }
    }
}

unsigned int flipBits(unsigned int num)
{
    return (((num >> 24) & 0xff) << 0) +
        (((num >> 16) & 0xff) << 8) +
        (((num >> 8) & 0xff) << 16) +
        (((num >> 8) & 0xff) << 24);

}

void loadMNIST(const char* fileNameImages, const char* fileNameLabels,
        double*** data, unsigned int* nData, unsigned int* nX, unsigned int* nY)
{
    FILE* fImage = fopen(fileNameImages, "rb");
    FILE* fLabel = fopen(fileNameLabels, "rb");

    unsigned int num;

    fread(&num, 4, 1, fImage);
    unsigned int magicNumber = flipBits(num);
    printf("Magic number for Images is: %u\n", magicNumber);

    fread(&num, 4, 1, fImage);
    unsigned int nImages = flipBits(num);
    printf("Number of images is: %u\n", nImages);

    fread(&num, 4, 1, fImage);
    unsigned int nRows = flipBits(num);
    printf("Number of rows is: %u\n", nRows);

    fread(&num, 4, 1, fImage);
    unsigned int nCols = flipBits(num);
    printf("Number of columns is: %u\n", nCols);

    fread(&num, 4, 1, fLabel);
    magicNumber = flipBits(num);
    printf("Magic number for Labels is: %u\n", magicNumber);

    fread(&num, 4, 1, fLabel);
    unsigned int nLabels = flipBits(num);
    printf("Number of labels is: %u\n", nLabels);

    // Now load all images:
    unsigned char v;
    if((*data) != NULL) {
        printf("Warning, potential memory leak\n");
        free((*data));
    }
    (*data) = (double**) malloc(nImages * sizeof((*data)));

    for (unsigned int i = 0; i < nImages; i++) {
        (*data)[i] = (double*) malloc((nRows * nCols + 10) * sizeof((*data)[i]));
        for (unsigned int j = 0; j < nRows * nCols; j++) {
            fread(&v, 1, 1, fImage);
            (*data)[i][j] = v/255.0;
        }
        fread(&v, 1, 1, fLabel);
        //printf("Label is: %d\n", v);
        for (unsigned int j = 0; j < 10; j++) { 
            (*data)[i][nRows * nCols + j] = double(j == v);
        }
        //printImage((*data)[i]);
    }

    fclose(fImage);
    fclose(fLabel);
    (*nData) = nImages;
    (*nX) = nRows * nCols;
    (*nY) = 10;

}

void backprop(CostFunction* costFunction,
        const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights,
        double* observation, unsigned int nX, unsigned int nY,
        double** delta_nabla_b, double*** delta_nabla_w)
{
    // allocating all necessary stuff
    //double** nabla_b = (double**) malloc((nLayer-1) * sizeof(nabla_b));
    //double*** nabla_w = (double***) malloc((nLayer-1) * sizeof(nabla_w));
    double** activations = (double**) malloc(nLayer * sizeof(activations)); 
    activations[0] = observation;
    double** zs = (double**) malloc((nLayer-1) * sizeof(zs));

    for (unsigned int i = 0; i < nLayer-1; i++) {

        //nabla_b[i] = (double*) malloc(sizes[i+1] * sizeof(nabla_b[i]));
        //nabla_w[i] = (double**) malloc(sizes[i+1] * sizeof(nabla_w[i]));
        zs[i] = (double*) calloc(sizes[i+1], sizeof(zs[i]));
        activations[i+1] = (double*) malloc(sizes[i+1] * sizeof(activations[i+1]));

        for (int j = 0; j < sizes[i+1]; j++) {
            //nabla_w[i][j] = (double*) malloc(sizes[i] * sizeof(nabla_w[i][j])); 

            for (int k = 0; k < sizes[i]; k++) {
                zs[i][j] += weights[i][j][k] * activations[i][k]; 
            }
            zs[i][j] += biases[i][j];

            activations[i+1][j] = sigmoid(zs[i][j]); 

        }

    }
    //for (int i = 0; i < nLayer-1; i++) {
    //    printf("Biases of layer %d:\n", i+1);
    //    for (int j = 0; j < sizes[i+1]; j++) {
    //        printf(" %f\n", biases[i][j]);
    //    }
    //}
    //for (int i = 0; i < 30; i++) {
    //    printf("activations[1][%d]: %f\n", i, activations[1][i]); 
    //}
    //for (int i = 0; i < 10; i++) {
    //    printf("activations[2][%d]: %f\n", i, activations[2][i]); 
    //}
   
    double* deltaPrev = (double*) malloc(sizes[nLayer-1] * sizeof(deltaPrev));
    for (int i = 0; i < sizes[nLayer-1]; i++) {
        //delta[i] = activations[nLayer-1][i] - observation[nX + i];
        deltaPrev[i] = costFunction->delta(sigmoidPrime(zs[nLayer-2][i]), activations[nLayer-1][i], (int) observation[nX + i]);
        //deltaPrev[i] *= sigmoidPrime(zs[nLayer-2][i]);
        delta_nabla_b[nLayer-2][i] = deltaPrev[i];
        for (int j = 0; j < sizes[nLayer-2]; j++) {
            delta_nabla_w[nLayer-2][i][j] = deltaPrev[i] * activations[nLayer-2][j];
        }
    } 

    for (unsigned int l = 1; l < nLayer-1; l++) {
        double* deltaNext = (double*) calloc(sizes[nLayer-1 -l], sizeof(deltaNext));
        for (int i = 0; i < sizes[nLayer-1 -l]; i++) {
            for (int j = 0; j < sizes[nLayer-1 - l + 1]; j++) {
                deltaNext[i] += weights[nLayer-2 -l + 1][j][i] * deltaPrev[j]; 
            }
            deltaNext[i] *= sigmoidPrime(zs[nLayer-2 -l][i]);
            delta_nabla_b[nLayer-2 -l][i] = deltaNext[i];
            for (int j = 0; j < sizes[nLayer-2 -l]; j++) { 
                delta_nabla_w[nLayer-2 -l][i][j] = deltaNext[i] * activations[nLayer-2 -l][j];
            }
        }
        free(deltaPrev);
        deltaPrev = deltaNext;
    }
    free(deltaPrev);
    deltaPrev = NULL;
    //for (int i = 0; i < sizes[1]; i++) {
    //    printf("delta_nabla_b[0][%d] = %f\n", i, delta_nabla_b[0][i]);
    //}
    //for (int i = 0; i < sizes[2]; i++) {
    //    printf("delta_nabla_b[1][%d] = %f\n", i, delta_nabla_b[1][i]);
    //}

    //for (int i = 0; i < nLayer-1; i++) {
    //    printf("Weights of layer %d:\n", i+1);
    //    for (int j = 0; j < sizes[i+1]; j++) {
    //        for (int k = 0; k < sizes[i]; k++) {
    //            printf(" %f ", delta_nabla_w[i][j][k]);
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //}
    
    //(*delta_nabla_b) = nabla_b;
    //(*delta_nabla_w) = nabla_w;


    
    // freeing everithing
    for (int i = 0; i < nLayer-1; i++) {
        free(activations[i+1]);
        free(zs[i]);
    }
    free(zs);
    free(activations);
}

void learnFromMiniBatch(CostFunction* costFunction,
        const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights,
        const unsigned int nData, double** data, unsigned int nX, unsigned int nY, unsigned int* index,
        unsigned int start, const unsigned int miniBatchSize, const double eta, const double lambda)
{

    // allocating all necessary stuff
    double** nabla_b = (double**) malloc((nLayer-1) * sizeof(nabla_b));
    double** delta_nabla_b = (double**) malloc((nLayer-1) * sizeof(delta_nabla_b));
    double*** nabla_w = (double***) malloc((nLayer-1) * sizeof(nabla_w));
    double*** delta_nabla_w = (double***) malloc((nLayer-1) * sizeof(delta_nabla_w));

    for (int i = 0; i < nLayer-1; i++) {
        nabla_b[i] = (double*) calloc(sizes[i+1], sizeof(nabla_b[i]));
        delta_nabla_b[i] = (double*) calloc(sizes[i+1], sizeof(delta_nabla_b[i]));
        nabla_w[i] = (double**) malloc(sizes[i+1] * sizeof(nabla_w[i]));
        delta_nabla_w[i] = (double**) malloc(sizes[i+1] * sizeof(delta_nabla_w[i]));
        for (int j = 0; j < sizes[i+1]; j++) {
            nabla_w[i][j] = (double*) calloc(sizes[i], sizeof(nabla_w[i][j])); 
            delta_nabla_w[i][j] = (double*) calloc(sizes[i], sizeof(delta_nabla_w[i][j])); 
        }
    }

    // now do the learning
    //printf("Now do the learning\n");
    //for (int i = 0; i < sizes[1]; i++) {
    //    printf("nabla_b[0][%d] = %f\n", i, nabla_b[0][i]);
    //}
    for (int i = start; i < start + miniBatchSize; i++) {
        backprop(costFunction, nLayer, sizes, biases, weights, data[index[i]], nX, nY, delta_nabla_b, delta_nabla_w);
        for (unsigned int layer = 0; layer < nLayer-1; layer++) {
            for (unsigned int j = 0; j < sizes[layer+1]; j++) {
                nabla_b[layer][j] += delta_nabla_b[layer][j]; 
                for (unsigned int k = 0; k < sizes[layer]; k++) {
                    nabla_w[layer][j][k] += delta_nabla_w[layer][j][k]; 
                }
            }
        }
    }

    for (unsigned int layer = 0; layer < nLayer-1; layer++) {
        for (unsigned int j = 0; j < sizes[layer+1]; j++) {
            biases[layer][j] -= eta/miniBatchSize * nabla_b[layer][j]; 
            for (unsigned int k = 0; k < sizes[layer]; k++) {
                weights[layer][j][k] *= (1.0 - eta * lambda / nData); 
                weights[layer][j][k] -= eta/miniBatchSize * nabla_w[layer][j][k]; 
            }
        }
    }

    // printing values
    //printf("after 1st call to learning\n");
    //for (int i = 0; i < nLayer-1; i++) {
    //    //printf("Biases of layer %d:\n", i+1);
    //    for (int j = 0; j < sizes[i+1]; j++) {
    //        
    //        printf(" %f\n", biases[i][j]);
    //    }
    //    printf("\n");
    //}
   


    // freeing everithing
    for (int i = 0; i < nLayer-1; i++) {
        for (int j = 0; j < sizes[i+1]; j++) {
            free(delta_nabla_w[i][j]);
            free(nabla_w[i][j]);
        }
        free(delta_nabla_w[i]);
        free(nabla_w[i]);
        free(delta_nabla_b[i]);
        free(nabla_b[i]);
    }
    free(delta_nabla_w);
    free(nabla_w);
    free(delta_nabla_b);
    free(nabla_b);

} 

unsigned int indexMax(double* array, unsigned int N) 
{
    //printf("Array has %u elements\n", N);
    unsigned int k = 0;
    double max = array[k];

    for (unsigned int i = 0; i < N; i++) {
        if (array[i] > max) {
            max = array[i];
            k = i;
        }
    }
    return k;
}

double evaluate(
        const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights,
        const unsigned int nTest, double** test)
{
    unsigned int success = 0;
    double* input = NULL;
    unsigned int trueValue;
    double* output = NULL;

    for (unsigned int i = 0; i < nTest; i++) {
        input = test[i];
        trueValue = indexMax(&test[i][sizes[0]], sizes[nLayer-1]);
        //printf("The unsigned int true value is: %u\n", trueValue);
        feedForward(nLayer, sizes, biases, weights, input, &output);
        
        // print result of feedForward
        //printf("result of feedforwarding:\n");
        //for (int j = 0; j < sizes[nLayer-1]; j++) {
        //    printf("%f\n", output[j]); 
        //}
        unsigned int prediction = indexMax(output, sizes[nLayer-1]);
        //printf("Prediction = %u, expected = %u\n", prediction, trueValue);
        if (prediction == trueValue) success++; 
        free(output);
        output = NULL;
    }
    return success*100./nTest;

}
int loadNetwork(const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights)
{
    char outputFileName[255];
    sprintf(outputFileName, "%u_", nLayer);
    char strSizes[15];
    for (unsigned int i = 0; i < nLayer; i++) {
        sprintf(strSizes, "%u_", sizes[i]);
        strcat(outputFileName, strSizes);
    }
    strcat(outputFileName, "nn.txt");

    FILE *f = fopen(outputFileName, "r");
    if (f == NULL) {
        printf("Error opening file!\n");
        return -1;
    }

    unsigned int readNLayer = 0;
    // printing values
    fscanf(f, "%u", &readNLayer);
    if (readNLayer != nLayer) {
        printf("Problem, differents NN architectures.");
        return -1;
    }
    unsigned int* readSizes = (unsigned int*) malloc(nLayer * sizeof(readSizes));
    for (unsigned int i = 0; i < nLayer; i++) {
        fscanf(f, "%u", &readSizes[i]);
        if (readSizes[i] != sizes[i]) {
            printf("Problem, differents NN architectures.");
            return -1;
        }
    }
    for (unsigned int i = 0; i < nLayer-1; i++) {
        //printf("Biases of layer %d:\n", i+1);
        for (unsigned int j = 0; j < sizes[i+1]; j++) {
            
            fscanf(f, "%lf", &biases[i][j]);
        }
    }

    for (unsigned int i = 0; i < nLayer-1; i++) {
        for (unsigned int j = 0; j < sizes[i+1]; j++) {
            for (unsigned int k = 0; k < sizes[i]; k++) {
                fscanf(f, "%lf", &weights[i][j][k]);
            }
        }
    }
    fclose(f);
    return 0;
}
void saveNetwork(const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights)
{
    char outputFileName[255];
    sprintf(outputFileName, "%u_", nLayer);
    char strSizes[15];
    for (unsigned int i = 0; i < nLayer; i++) {
        sprintf(strSizes, "%u_", sizes[i]);
        strcat(outputFileName, strSizes);
    }
    strcat(outputFileName, "nn.txt");

    FILE *f = fopen(outputFileName, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    // printing values
    fprintf(f, "%u ", nLayer);
    for (unsigned int i = 0; i < nLayer; i++) {
        fprintf(f, "%u ", sizes[i]);
    }
    fprintf(f, "\n");
    fprintf(f, "\n");
    for (unsigned int i = 0; i < nLayer-1; i++) {
        //printf("Biases of layer %d:\n", i+1);
        for (unsigned int j = 0; j < sizes[i+1]; j++) {
            
            fprintf(f, "%.6f ", biases[i][j]);
        }
        fprintf(f, "\n");
        fprintf(f, "\n");
    }

    for (unsigned int i = 0; i < nLayer-1; i++) {
        for (unsigned int j = 0; j < sizes[i+1]; j++) {
            for (unsigned int k = 0; k < sizes[i]; k++) {
                fprintf(f, "%.6f ", weights[i][j][k]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    
}
void stochasticGradientDescent(CostFunction* costFunction,
        const unsigned int nLayer, const unsigned int* sizes, double** biases, double*** weights,
        const unsigned int nEpochs, const unsigned int miniBatchSize,
        const unsigned int nTraining, double** training, const unsigned int nTest, double** test,
        const unsigned int nX, const unsigned int nY, const double eta, const double lambda,
        const bool finalTestOnly)
{

    unsigned int* index = (unsigned int*) malloc(nTraining * sizeof(index));
    for (int i = 0; i < nTraining; i++) {
        index[i] = i; 
    }

    for (unsigned int epoch = 0; epoch < nEpochs; epoch++) {
        printf("Starting epoch %u\n", epoch);
        shuffle(index, nTraining); 

        for (unsigned int k = 0; k < nTraining; k += miniBatchSize) {
            //printf("Learning from miniBatch %u\n", k);
            //printImage(training[0]);
            learnFromMiniBatch(costFunction, nLayer, sizes, biases, weights, nTraining, training, nX, nY, index, k, miniBatchSize, eta, lambda); 
        } 

        //for (int i = 0; i < nLayer-1; i++) {
        //    printf("Biases of layer %d:\n", i+1);
        //    for (int j = 0; j < sizes[i+1]; j++) {
        //        printf(" %f\n", biases[i][j]);
        //    }
        //}
        //printf("\n");
        if (!finalTestOnly) {
            double epochSuccess = evaluate(nLayer, sizes, biases, weights, nTest, test);
            printf("Epoch %u done with success percentage of: %f\n", epoch, epochSuccess);
        }
        else {
            printf("Epoch %u done.\n", epoch);
        }
    } 
    if(finalTestOnly) {
        double epochSuccess = evaluate(nLayer, sizes, biases, weights, nTest, test);
        printf("Success after training: %.2f%%\n", epochSuccess);
    }

} 


int main()
{

    //srand(time(NULL));
    srand(1234);

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    const unsigned int nLayer = 3;
    const unsigned int sizes[nLayer] = {784, 100, 10};

    // allocating biases list of vectors and weights list of matrices
    double* input = (double*) calloc(sizes[0], sizeof(input));
    double* output = NULL;
    double** biases = (double**) malloc((nLayer-1) * sizeof(biases));
    double*** weights = (double***) malloc((nLayer-1) * sizeof(weights));

    for (int i = 0; i < nLayer-1; i++) {
        biases[i] = (double*) calloc(sizes[i+1], sizeof(biases[i]));
        weights[i] = (double**) malloc(sizes[i+1] * sizeof(weights[i]));
        for (int j = 0; j < sizes[i+1]; j++) {
            weights[i][j] = (double*) calloc(sizes[i], sizeof(weights[i][j])); 
        }
    }

    
    int networkLoaded = loadNetwork(nLayer, sizes, biases, weights);
    // printing values
    if (networkLoaded != 0) {
        for (int i = 0; i < nLayer-1; i++) {
            //printf("Biases of layer %d:\n", i+1);
            for (int j = 0; j < sizes[i+1]; j++) {
                
                biases[i][j] = rand() * 2.0 /RAND_MAX - 1;
                printf(" %f\n", biases[i][j]);
            }
            printf("\n");
        }

        for (int i = 0; i < nLayer-1; i++) {
            printf("Weights of layer %d:\n", i+1);
            for (int j = 0; j < sizes[i+1]; j++) {
                for (int k = 0; k < sizes[i]; k++) {
                    weights[i][j][k] = rand() * 2.0 /RAND_MAX - 1;
                    printf(" %f ", weights[i][j][k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    //// create input for test
    //for (int i = 0; i < sizes[0]; i++) {
    //    input[i] = 1.0;
    //}

    //// feedForward function
    //printf("Ready for feedForward\n");
    //feedForward(nLayer, sizes, biases, weights, input, &output);

    //// print result of feedForward
    //printf("result of feedforwarding:\n");
    //for (int i = 0; i < sizes[nLayer-1]; i++) {
    //    printf("%f\n", output[i]); 
    //}

    CostFunction quadratic = {quadraticCostFn, quadraticCostDelta};
    CostFunction xEntropy = {xEntropyCostFn, xEntropyCostDelta};
    
    const char* fileNameTrainingImages = "train-images.idx3-ubyte";
    const char* fileNameTrainingLabels = "train-labels.idx1-ubyte";
    const char* fileNameTestImages = "t10k-images.idx3-ubyte";
    const char* fileNameTestLabels = "t10k-labels.idx1-ubyte";

    double** trainingData = NULL;
    double** testData = NULL;
    unsigned int nTraining, nTest;
    unsigned int nTrainingX, nTestX, nTrainingY, nTestY;

    loadMNIST(fileNameTrainingImages, fileNameTrainingLabels, &trainingData, &nTraining, &nTrainingX, &nTrainingY);
    loadMNIST(fileNameTestImages, fileNameTestLabels, &testData, &nTest, &nTestX, &nTestY);
    
    const unsigned int nEpochs = 60;
    const unsigned int miniBatchSize = 10;
    const double eta = 0.1;
    const double lambda = 5.0;
    stochasticGradientDescent(&xEntropy, nLayer, sizes, biases, weights, nEpochs, miniBatchSize, nTraining, trainingData, nTest, testData, nTrainingX, nTrainingY, eta, lambda, false);
    saveNetwork(nLayer, sizes, biases, weights);
    
    
    // freeing biases list of vectors and weights list of matrices
    printf("Freeing everything\n");
    for (int i = 0; i < nLayer-1; i++) {
        for (int j = 0; j < sizes[i+1]; j++) {
            free(weights[i][j]);
        }
        free(weights[i]);
        free(biases[i]);
    }
    free(weights);
    free(biases);
    free(output);
    free(input);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f\n", cpu_time_used);


}

