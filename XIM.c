//XOR in the framework
#include "inc/Nero.h"


float train[] = {
	0, 0, 1,
	1, 0, 0,
	0, 1, 0,
	1, 1, 1,
};


int main(void){
	srand(time(0));

    size_t arch[] = {2,2,1};
    
    NN_Model model = NN_ALLOC(arch,array_len(arch));

    NN_rand(model,0,1);




	size_t stride = 3;
	size_t n = sizeof(train)/sizeof(train[0])/stride;

	Mat ti = Mat_cut(train,n,2,3,0);	
	Mat to = Mat_cut(train,n,1,3,2);	

    Mat row;
    for(size_t i = 0 ;i < 4 ; i++){
        row = Mat_row(ti,i);
        Mat_copy(NN_INPUT(model),row);
        NN_feedforward(model);
        Mat_SHOW(NN_OUTPUT(model));
        NN_print(model,"XOR");
    }


    return 0;
}
