#include "../inc/Nero.h"
#define NN_IMPLI
#include "mnist.h"
/*  from mnist inclution
    train image : train_image[60000][784] (type: double, normalized, flattened)
    train label : train_label[60000] (type: int)
    test image : test_image[10000][784] (type: double, normalized, flattened)
    test label : test_label[10000] (type: int)
*/
float train_label_prep[60000][10] = {0};
//nasty code fix later future me 

int main(void){
    srand(69);
    load_mnist();

    //getting the dataset into an acceptable format
    //This is the "form" of the neural network
	size_t arch[] = {784,10,10,10};
    //generating the model and its gradient
    NN_Model model = NN_ALLOC(arch,array_len(arch));
    NN_Model gradi = NN_ALLOC(arch,array_len(arch));
    //get the rand model 
    NN_rand(model,-1,1);
    //describing the input data

    Mat ti = {
        .rows = 60000,
        .cols = 784,
        .stride = 784,
        .ptr = &train_image[0][0],
    };

    Mat_STAT(ti);

    //prepss the output matrix of the model
    for(size_t i = 0 ; i < 60000 ; i++){
        size_t temp = (size_t)train_label[i];
        train_label_prep[i][temp] = 1.f;
    } 

    Mat to = {
        .rows = 60000,
        .cols = 10,
        .stride = 10,
        .ptr = &train_label_prep[0][0],
    };

    Mat_STAT(to);
    printf("%f \n",NN_cost(model,ti,to,60));
    /*for(size_t i = 0 ; i < 10 ; i++){
        //NN_finit_diff(model,gradient,ti,to,n,1);
        NN_backprop(model,gradi,ti,to);
        NN_gradient_update(model,gradi,1);
        printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,60000));
    }*/

    return 0;
}
