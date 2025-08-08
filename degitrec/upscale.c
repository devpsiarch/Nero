#include "../inc/Nero.h"
#include "../inc/see.h"
#include "./mnist.h"

// draw a gray scale picture given a matrix 
void show_picture(Mat mat,const int w,const int h,const int size){
    InitWindow(w,h,"Showing picture function");

    while (!WindowShouldClose())
    {
        BeginDrawing();
            ClearBackground(WHITE);
            for(size_t i = 0 ; i < mat.rows ; i++){
                for(size_t j = 0 ; j < mat.cols ; ++j){
                    float val = Mat_at(mat,i,j);
                    Color c = {255*val,255*val,255*val,255};
                    DrawRectangle(i*size,j*size,size,size,c);
                }
            } 
        EndDrawing();
    }

    CloseWindow();
}


int main(void){
    load_mnist();
    srand(time(0));
    
    //size_t a[] = {2,1,1};
    // NN_Model test = NN_ALLOC(a,array_len(a));
    // Mat_fill(test.wi[0],69.f);
    // Mat_fill(test.wi[1],420.f);
    // Mat_fill(test.bi[0],-1.f);
    // Mat_fill(test.bi[1],1.f);
    // NN_SAVE(test,"saved.nn");
    // NN_FREE(test);

    NN_Model loaded = NN_LOAD("saved.nn");
    NN_print(loaded,"loaded model");
    printf("size of model %zu\n",NN_SIZE(loaded));
    NN_FREE(loaded);
   
    return 0; 

    size_t arch[] = {2,10,1};
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);
    
    const size_t index = 0;

    const float rate = 2.0f;
    const size_t epochs = 1*1000;
    const size_t size = 28;

    Mat t = Mat_alloc(size*size,3);
    for(size_t i = 0 ; i < size ; ++i){
        for(size_t j = 0 ; j < size ; ++j){
            Mat_at(t,i*size+j,0) = (float)j/((float)size-1);
            Mat_at(t,i*size+j,1) = (float)i/((float)size-1);
            Mat_at(t,i*size+j,2) = train_image[index][i*size+j];
        }
    }
    // Mat_SHOW(t);

    Mat ti = {
        .rows = size*size,
        .cols = 2,
        .stride = 3,
        .ptr = &Mat_at(t,0,0)
    };
     Mat to = {
        .rows = size*size,
        .cols = 1,
        .stride = 3,
        .ptr = &Mat_at(t,0,2)
    };

    // Mat_SHOW(ti);
    // Mat_SHOW(to);

    // training / memorizing the picture
    for(size_t i = 0 ; i < epochs ; ++i){
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,rate);    
        if(i % 100 == 0) printf("INFO: %zu - cost : %f \n",i,NN_cost(model,ti,to,size*size));
    }


    // generating an image 
    Mat generated = {
        .rows = size,
        .cols = size,
        .stride = size,
        .ptr = malloc(sizeof(float)*size*size)
    };

    for(size_t i = 0 ; i < size ; ++i){
        for(size_t j = 0 ; j < size ; ++j){
            Mat_at(NN_INPUT(model),0,0) = (float)j/((float)size-1);
            Mat_at(NN_INPUT(model),0,1) = (float)i/((float)size-1);
            NN_feedforward(model);
            Mat_at(generated,i,j) = Mat_at(NN_OUTPUT(model),0,0); 
        }
    }

    // show_picture(generated,800,450,10);

    // Mat_SHOW(og);
    // Mat_SHOW(generated);

defer:
    NN_FREE(model);
    NN_FREE(gradient);
    Mat_free(t);
    Mat_free(generated);
    return 0;
}
