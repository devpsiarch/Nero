//TODO => cost -> sum matrix 
//TODO ==> maybe we can group the gradient and the model into one datastruct ? 
//TODO ==> we gotta make a methode of parsing "collecting" any data and orginizing it into a scheme the API can handel ...
//"1D array" , Or maybe there exists external libraries ...
//TODO ==> Maybe we can group the "arch" and the stride into one datastruct so it can be less annoying  
//TODO ==> maybe we can abstract the API even more ???? 

#include "inc/Nero.h"
#include "inc/see.h"
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
    srand(time(0));
    const int cell_size = 10;
    const int hight = 450;
    const int width = 800;

    size_t arch[] = {6,5,10,3};
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);
    
    size_t stride = 7;
	  size_t n = train_rows(raw_train,stride);
    
    // prep a raw pointer for the data ... ik amazing api
    float *train = (float*)malloc(sizeof(float)*n*stride);
    copy(train,raw_train,n*stride);  

    Mat ti = Mat_cut(train,n,arch[0],stride,0);
	  Mat to = Mat_cut(train,n,last_element(arch),stride,to_get_offset(stride,last_element(arch)));	

    // pre visulizer
    NN_map* map =  init_nnmap(arch,array_len(arch),width,hight);

    InitWindow(width,hight, "Brain show me");
    while (!WindowShouldClose()){
        BeginDrawing();
            ClearBackground(BLACK);
            NN_backprop(model,gradient,ti,to);
            NN_gradient_update(model,gradient,0.1);    
            draw_nn(model,map,arch,array_len(arch));
        EndDrawing();
    }

    CloseWindow();
    printf("Terminal | cost : %f \n",NN_cost(model,ti,to,n));
    NN_print(model,"adder"); 
defer:
    NN_FREE(model);
    NN_FREE(gradient);
    free(train);
    kill_nnmap(map);
    return 0;
}
