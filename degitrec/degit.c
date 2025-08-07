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
                    DrawRectangle(j*size,i*size,size,size,c);
                }
            } 
        EndDrawing();
    }

    CloseWindow();
}


int main(void){
    load_mnist();
    //print_mnist_pixel(train_image,train_label,1);

    srand(69);

    size_t arch[] = {2,5,3,1};
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,-10,10);

    const size_t rows_treated = 1;
    const size_t index = 0;


    Mat picture = {
        .rows = 28,
        .cols = 28,
        .stride = 28,
        .ptr = &train_image[index][0],
    };


    Mat ti = {
        .rows = 28*28,
        .cols = 2,
        .stride = 2,
        .ptr = malloc(sizeof(float)*2*28*28),
    };
   
    for(size_t i = 0 ; i < 28 ; i++){
        for(size_t j = 0 ; j < 28 ; j++){
            Mat_at(ti,i*28+j,0) = i;
            Mat_at(ti,i*28+j,1) = j;
        }
    }

    Mat to = {
        .rows = 28*28,
        .cols = 1,
        .stride = 1,
        .ptr = malloc(sizeof(float)*28*28),
    };

    show_picture(picture,800,450,10);


    for(size_t i = 0 ; i < 28*28 ; i++){
        Mat_at(to,i,0) = train_image[index][i];
    }
    // Mat_SHOW(ti);
    // Mat_SHOW(to);
    
    //Mat_SHOW(picture);

    // memorizing the picture
    for(size_t i = 0 ; i < 10000 ; ++i){
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,1.5);    
        printf("Terminal | cost : %f \n",NN_cost(model,ti,to,rows_treated));
    }
    // finding the difference between the 'picture' and the 'generated' one
    // its the same as the cost function result 


    const size_t new_size = 28;
    const float scale = new_size/28.f;

    Mat generated = {
        .rows = new_size,
        .cols = new_size,
        .stride = new_size,
        .ptr = malloc(sizeof(float)*new_size*new_size),
    };


    for(size_t i = 0 ; i < generated.rows ; ++i){
        for(size_t j = 0 ; j < generated.cols ; ++j){
            float values[2] = {(float)i*scale,(float)j*scale};
                Mat input = {
                    .rows = 1,
                    .cols = 2,
                    .stride = 2,
                    .ptr = &values[0],
                };
            NN_PREDICT(input,model);
            // Mat_at(generated,i,j) = Mat_at(model.ai[model.layers],0,0);
        }
    }

    show_picture(generated,850,450,15);

    //Mat_SHOW(generated);

    // we can now "in a sense" compress the 28*28 picture to 
    // a neural network, we can generate the same image again
    // but what more intresting is upscalling the image 
    // since the inputs of the network are continous

defer:
    NN_FREE(model);
    NN_FREE(gradient);
    Mat_free(to);
    Mat_free(ti);
    Mat_free(generated);
    return 0;
}
