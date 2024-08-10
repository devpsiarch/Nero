//TODO => cost -> sum matrix 

//XOR in the framework
#include "inc/Nero.h"


float train[] = {
	0, 0, 0,
	1, 0, 1,
	0, 1, 1,
	1, 1, 0,
};


int main(void){
	srand(time(0));

    size_t arch[] = {2,4,1};
    
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);




	size_t stride = 3;
	size_t n = sizeof(train)/sizeof(train[0])/stride;

	Mat ti = Mat_cut(train,n,2,3,0);	
	Mat to = Mat_cut(train,n,1,3,2);	

    Mat row;
    for(size_t i = 0 ;i < 4 ; i++){
        row = Mat_row(ti,i);
        Mat_STAT(row);
        Mat_copy(NN_INPUT(model),row);
        NN_feedforward(model);
        Mat_SHOW(NN_OUTPUT(model));
        NN_print(model,"XOR");
    }


    printf("%f \n",NN_cost(model,ti,to,4));
        
    for(size_t i = 0 ; i < 100*1000 ; i++){
        NN_finit_diff(model,gradient,ti,to,4,1e-1);
        NN_gradient_update(model,gradient,1e-1);
    }

    printf("terminal cost is : %f \n",NN_cost(model,ti,to,4));

    return 0;
}
