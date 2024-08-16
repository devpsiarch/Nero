#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "inc/Nero.h"

#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define SIZE 784

int train_label[NUM_TRAIN];
int test_label[NUM_TEST];

float train_image[NUM_TRAIN][SIZE];
float test_image[NUM_TEST][SIZE];

// define
#define PACKED __attribute__((packed))

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define MAP_MNIST(i) ((((i)&0xFF000000) >> 24) | (((i)&0x00FF0000) >> 8) | (((i)&0x0000FF00) << 8) | (((i)&0x000000FF) << 24))
#else
#define MAP_MNIST(i) (i)
#endif

#define IMAGES_MNIST 2051 // According to Mnist
#define LABELS_MNIST 2049 // According to Mnist



// struct
struct ExpectedHeaderStr
{
    uint32_t type; // magic number : need to be 2049
    uint32_t size;
} PACKED;
typedef struct ExpectedHeaderStr ExpectedHeader;

struct ImageHeaderStr
{
    uint32_t type; // magic number : need to be 2051
    uint32_t size;
    uint32_t height; // nb of rows in each image
    uint32_t width;  // nb of columns in each image
} PACKED;
typedef struct ImageHeaderStr ImageHeader;


// code for the labels
void labels(const char *path,int type)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    ExpectedHeader labelHeader;
    if (fread(&labelHeader, sizeof(ExpectedHeader), 1, file) != 1)
    {
        fprintf(stderr, "Could not read label header from: %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    labelHeader.type = MAP_MNIST(labelHeader.type);
    labelHeader.size = MAP_MNIST(labelHeader.size);

    if (labelHeader.type == LABELS_MNIST)
    {
        uint8_t *labels = (uint8_t *)malloc(sizeof(uint8_t) * labelHeader.size);
        if (fread(labels, sizeof(uint8_t), labelHeader.size, file) != labelHeader.size)
        {
            fprintf(stderr, "Could not read %u labels\n", labelHeader.size);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        if (type == 0) // train
        {
            for (size_t i = 0; i < NUM_TRAIN; i++)
            {
                train_label[i] = (int)labels[i];
            }
        }else if (type == 1)
        {
            for (size_t i = 0; i < NUM_TEST; i++)
            {
                test_label[i] = (int)labels[i];
            }
        }else printf("no valid dst\n");
        
        // labels should be yours labels
        free(labels);
    }

    
    fclose(file);
}

// code for the images
void images(const char *path,size_t space)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Could not open file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    ImageHeader imageHeader;
    if (fread(&imageHeader, sizeof(ImageHeader), 1, file) != 1)
    {
        fprintf(stderr, "Could not read label header from: %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    imageHeader.type = MAP_MNIST(imageHeader.type);
    imageHeader.size = MAP_MNIST(imageHeader.size);
    imageHeader.height = MAP_MNIST(imageHeader.height);
    imageHeader.width = MAP_MNIST(imageHeader.width);
    size_t size = imageHeader.size;
    size_t sizeValue = imageHeader.height * imageHeader.width;
    if (size != space /*check the size of labels*/)
    {
        fprintf(stderr, "Number of images does not match number of labels (%zu != %d)\n", (size_t)size, 0 /*check the siwe of labels*/);
        exit(EXIT_FAILURE);
    }

    if (imageHeader.type == IMAGES_MNIST)
    {
        size_t fullData = sizeValue * size;
        uint8_t *data = (uint8_t*)malloc(sizeof(uint8_t) * fullData);
        if (fread(data, sizeof(uint8_t), fullData, file) != fullData)
        {
            fprintf(stderr, "Could not read %zu images\n", size);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        float **value = (float**)malloc(sizeof(float *) * size);
        for (size_t i = 0; i < size; ++i)
        {
            value[i] = (float*)malloc(sizeof(float) * sizeValue);
            size_t offset = i * sizeValue;
            for (size_t k = 0; k < sizeValue; ++k)
            {
                value[i][k] = ((float)data[k + offset] / 255);
            }
        }
        free(data);
        if (space == NUM_TRAIN)
        {
            for (size_t i = 0; i < NUM_TRAIN; i++)
            {
                for (size_t j = 0; j < 784; j++)
                {
                    train_image[i][j] = value[i][j];
                }
                
            } 
        }else if (space == NUM_TEST)
        {
            for (size_t i = 0; i < NUM_TEST; i++)
            {
                for (size_t j = 0; j < 784; j++)
                {
                    test_image[i][j] = value[i][j];
                }
                
            }
        }
        // value should be yours images
        free(value);
    }

    fclose(file);
}

void print_mnist_pixel(float data_image[][SIZE],int label[], int num_data)
{
    int i, j;
    for (i=0; i<num_data; i++) {
        printf("image %d/%d\n", i+1, num_data);
        for (j=0; j<SIZE; j++) {
            if(data_image[i][j] != 0.f){
				printf(ANSI_COLOR_RED"%1.1f "ANSI_COLOR_RESET, data_image[i][j]);
			}else{
				printf("%1.1f ", data_image[i][j]);
			}
            if ((j+1) % 28 == 0) putchar('\n');
        }
        printf("%d",label[i]);
        putchar('\n');
    }
}

void load_mnist(void){
    labels("data/train-labels.idx1-ubyte",0);
    labels("data/t10k-labels.idx1-ubyte",1);

    images("data/train-images.idx3-ubyte",60000);
    images("data/t10k-images.idx3-ubyte",10000);
}
