#include "inc/Nero.h"
#include "mnist.h"


float train[] = {
	0, 0   ,0,0,   0,0,0,
	0, 0   ,0,1,   0,1,0,
	0, 0   ,1,0,   1,0,0,
	0, 0   ,1,1,   1,1,0, 
	0, 1   ,0,0,   0,1,0,
	0, 1   ,0,1,   1,0,0,
	0, 1   ,1,0,   1,1,0,
	0, 1   ,1,1,   0,0,1, 
	1, 0   ,0,0,   1,0,0,
	1, 0   ,0,1,   1,1,0, 
	1, 0   ,1,0,   0,0,1, 
	1, 0   ,1,1,   0,0,1,
	1, 1   ,0,0,   0,0,1, 
	1, 1   ,0,1,   0,0,1, 
	1, 1   ,1,0,   0,0,1,
	1, 1   ,1,1,   0,0,1, 
};

float train_mnist[60000*785];

void check_train(){
    size_t i ,j;
    for (i=0; i<60000; i++) {
        for (j=0; j<=SIZE; j++) {
            size_t loc = i*SIZE+j;
            if(j == SIZE){
                printf(ANSI_COLOR_GREEN"%f "ANSI_COLOR_RESET, train_mnist[loc+1]);
                continue;
            }
            if(train_mnist[loc] != 0.f){
				printf(ANSI_COLOR_RED"%1.1f "ANSI_COLOR_RESET, train_mnist[loc]);
			}else{
				printf("%1.1f ", train_mnist[loc]);
			}
            if ((j+1) % 28 == 0){
                putchar('\n');
            }

        }
        putchar('\n');
    }
}

void extract(){
    for (size_t i = 0 ; i < 60000; i++) {
        for (size_t j = 0 ; j <= 784 ;j++) {
            size_t loc = i*SIZE+j;
            train_mnist[loc] = train_image[i][j];
        }
    }
}

int main(void){
   
    load_mnist();
    extract();



    check_train(); 

    return 0;
    
    Mat test = {
        .rows = 60000,
        .cols = SIZE,
        .ptr = train_mnist,
    };
    

    return 0;
    
    srand(69);
    
    size_t arch[] = {4,5,3};
    
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);


	size_t stride = 7;
	size_t n = train_size(train,stride);
    
    //preps the tito
	Mat ti = Mat_cut(train,n,arch[0],stride,0);
	Mat to = Mat_cut(train,n,last_element(arch),stride,to_get_offset(stride,last_element(arch)));	


    Mat_SHOW(ti);
    Mat_SHOW(to);

    return 0;

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
        
    for(size_t i = 0 ; i < 10 ; i++){
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
