//TODO => cost -> sum matrix 
//TODO ==> maybe we can group the gradient and the model into one datastruct ? 
//TODO ==> we gotta make a methode of parsing "collecting" any data and orginizing it into a scheme the API can handel ...
//"1D array" , Or maybe there exists external libraries ...
//TODO ==> Maybe we can group the "arch" and the stride into one datastruct so it can be less annoying  
//TODO ==> maybe we can abstract the API even more ???? 

#include "inc/Nero.h"
// This nerual network is basically ADDER cercuit !!!

float raw_train[] = {
    0, 0, 0, 0,   0, 0, 0,  // Sum = 0 (000)
    0, 0, 0, 1,   0, 0, 1,  // Sum = 1 (001)
    0, 0, 1, 0,   0, 1, 0,  // Sum = 1 (001)
    0, 0, 1, 1,   0, 1, 1,  // Sum = 2 (010)
    0, 1, 0, 0,   0, 0, 1,  // Sum = 1 (001)
    0, 1, 0, 1,   0, 1, 0,  // Sum = 2 (010)
    0, 1, 1, 0,   0, 1, 1,  // Sum = 2 (010)
    0, 1, 1, 1,   1, 0, 0,  // Sum = 3 (011)
    1, 0, 0, 0,   0, 1, 0,  // Sum = 1 (001)
    1, 0, 0, 1,   1, 1, 0,  // Sum = 2 (010)
    1, 0, 1, 0,   0, 0, 1,  // Sum = 2 (010)
    1, 0, 1, 1,   0, 0, 1,  // Sum = 3 (011)
    1, 1, 0, 0,   0, 0, 1,  // Sum = 2 (010)
    1, 1, 0, 1,   0, 0, 1,  // Sum = 3 (011)
    1, 1, 1, 0,   0, 0, 1,  // Sum = 3 (011)
    1, 1, 1, 1,   1, 1, 0,  // Sum = 4 (100)
};

void copy(float* data,float train[],size_t size){
    for(size_t i = 0 ; i < size ;i++){
        data[i] = train[i]; 
    }
}


int main(void){
	srand(69);

    size_t arch[] = {4,10,10,3};
    
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);


    //n represents the training sets rows and 
    // stride is the datas reach or master cols in a sense
	size_t stride = 7;
	size_t n = train_rows(raw_train,stride);
    float *train = (float*)malloc(sizeof(float)*n*stride);
    copy(train,raw_train,n*stride);

    //Creating and the input and output matrix by using the descpription of "arch" and the stride 
    Mat ti = Mat_cut(train,n,arch[0],stride,0);
	Mat to = Mat_cut(train,n,last_element(arch),stride,to_get_offset(stride,last_element(arch)));	


    Mat_SHOW(ti);
    Mat_SHOW(to);

    Mat_STAT(ti);
    Mat_STAT(to);


/*
    Mat row;
    for(size_t i = 0 ;i < n ; i++){
        row = Mat_row(ti,i);
        //Mat_STAT(row);
        Mat_copy(NN_INPUT(model),row);
        NN_feedforward(model);
        //Mat_SHOW(NN_OUTPUT(model));
        //NN_print(gradient,"gradient");

    }
*/
    printf("%f \n",NN_cost(model,ti,to,n));
        
    for(size_t i = 0 ; i < 10000 ; i++){
        //NN_finit_diff(model,gradient,ti,to,n,1);
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,1);
        printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,n));
    }
    NN_print(gradient,"gradient after trainig process");
    
    printf("terminal cost is : %f \n",NN_cost(model,ti,to,n));

    printf("checking ...\n");

    NN_check(model,ti,n);
    
    NN_FREE(model);
    NN_FREE(gradient);
    
    return 0;
}
