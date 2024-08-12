//TODO => cost -> sum matrix 

//XOR in the framework
#include "inc/Nero.h"


#define train_size(train,stride) sizeof((train))/sizeof((train[0]))/(stride)

#define last_element(array) (array)[array_len((array))-1]

float train[] = {
	0, 0   ,0,0,   0,0,
	1, 0   ,1,0,   0,0,
	0, 1   ,1,0,   1,1,  
	1, 1   ,1,0,   0,1, 

};


int main(void){
	srand(69);

    size_t arch[] = {4,2,3,3,2};
    
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);


	size_t stride = 6;
	size_t n = train_size(train,stride);
    
    //preps the tito
	Mat ti = Mat_cut(train,n,arch[0],stride,0);
	Mat to = Mat_cut(train,n,last_element(arch),stride,stride-last_element(arch));	


    Mat_SHOW(ti);
    Mat_SHOW(to);


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
        
    for(size_t i = 0 ; i < 1000*1000 ; i++){
        //NN_finit_diff(model,gradient,ti,to,n,1);
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,1);
        //printf("cost : %f \n",NN_cost(model,ti,to,n));
    }
    NN_print(gradient,"gradient");
    
    printf("terminal cost is : %f \n",NN_cost(model,ti,to,n));

    printf("checking ...\n");

    NN_check(model,ti,n);

    NN_FREE(model);
    NN_FREE(gradient);
    
    return 0;
}
