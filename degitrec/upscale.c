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


void cmp_picture(Mat mat1,Mat mat2,const int w,const int h,const int size1,const int size2){
    InitWindow(w,h,"Compairing upscalled function");

    while (!WindowShouldClose())
    {
        BeginDrawing();
            ClearBackground(WHITE);
            for(size_t i = 0 ; i < mat1.rows ; i++){
                for(size_t j = 0 ; j < mat1.cols ; ++j){
                    float val = Mat_at(mat1,i,j);
                    Color c = {255*val,255*val,255*val,255};
                    DrawRectangle(i*size1,j*size1,size1,size1,c);
                }
            }
            for(size_t i = 0 ; i < mat2.rows ; i++){
                for(size_t j = 0 ; j < mat2.cols ; ++j){
                    float val = Mat_at(mat2,i,j);
                    Color c = {255*val,255*val,255*val,255};
                    DrawRectangle(i*size2+w/2,j*size2,size2,size2,c);
                }
            }
        EndDrawing();
    }

    CloseWindow();
}

int main(void){
    load_mnist();
    srand(70);

    const size_t size = 200;
    const float scale = (float)size/28.f; 
    NN_Model m = NN_LOAD("upscaler_5_index_0.nn");

    // generating an image 
    Mat upscalled = {
        .rows = size,
        .cols = size,
        .stride = size,
        .ptr = malloc(sizeof(float)*size*size)
    };
    Mat generated = {
        .rows = 28,
        .cols = 28,
        .stride = 28,
        .ptr = malloc(sizeof(float)*28*28)
    };
    for(size_t i = 0 ; i < size ; ++i){
        for(size_t j = 0 ; j < size ; ++j){
            Mat_at(NN_INPUT(m),0,0) = (float)j/((float)size-1);
            Mat_at(NN_INPUT(m),0,1) = (float)i/((float)size-1);
            NN_feedforward(m);
            Mat_at(upscalled,i,j) = Mat_at(NN_OUTPUT(m),0,0); 
        }
    }
    for(size_t i = 0 ; i < 28 ; ++i){
        for(size_t j = 0 ; j < 28 ; ++j){
            Mat_at(NN_INPUT(m),0,0) = (float)j/((float)28-1);
            Mat_at(NN_INPUT(m),0,1) = (float)i/((float)28-1);
            NN_feedforward(m);
            Mat_at(generated,i,j) = Mat_at(NN_OUTPUT(m),0,0); 
        }
    }
    cmp_picture(generated,upscalled,800,450,10,280.f/(float)size+1);

    NN_FREE(m); 
    Mat_free(generated);
    Mat_free(upscalled);
    return 0;

    size_t arch[] = {2,15,1};
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);
    
    const size_t index = 0;

    const float rate = 6.3f;
    const size_t epochs = 1000*1000;

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
        if(i % 1000 == 0) printf("INFO: %zu - cost : %f \n",i,NN_cost(model,ti,to,size*size));
    }


    

    //show_picture(generated,800,450,10);

    // we save the model
    NN_SAVE(model,"upscaler_5_index_0.nn");
    printf("size of model %zu bytes\n",NN_SIZE(model));
    
defer:
    NN_FREE(model);
    NN_FREE(gradient);
    Mat_free(t);
    Mat_free(generated);
    return 0;
}
