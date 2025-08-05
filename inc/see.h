#ifndef SEE_H
#define SEE_H
/*=============================*/
    /*visiulizing the model*/
/*=============================*/
#include "Nero.h"
#include <raylib.h>
//i dont wanna pass around these values everywhere
//maybe when i make a better API
#define WIDTH 1000
#define HIGHT 700
#define NEU_SIZE 50

size_t get_cubes(NN_Model model,size_t network_size);
void draw_neu(int num,int posX);
void draw_layers(size_t arr[],size_t size);

void draw_mat(Mat m,int posX,int posY);
void draw_model(NN_Model model);




#endif // !SEE_H
#ifndef SEE_IMPLI
#define SEE_IMPLI

/*=============================*/
    /*visiulizing the model*/
/*=============================*/
//maybe we can make two ways of visiulizing the learning process
//the neurons one is waaay to hard so ill try to make it later 
//on the other hand , the grid way is also promisingly easy ?


//maybe we can draw them like the neurons just in order 
//and save the locations after we split the canvas depending on 
//the the number of cubes , then we need a function that draw 
//thae the matrix not the model
size_t get_cubes(NN_Model model,size_t network_size){
    size_t res = 0;
    for(size_t i = 1 ; i < network_size ; i++){
        res += model.wi[i-1].rows*model.wi[i-1].cols;
        res += model.bi[i-1].rows*model.bi[i-1].cols;
    }
    return res;
}
void draw_mat(Mat m,int posX,int posY){
    for(size_t i = 0 ; i < m.rows; i++){
        for(size_t j = 0 ; j < m.cols ;j++){
            int x = posX+i*NEU_SIZE;
            int y = posY+j*NEU_SIZE;
            
            float val = Mat_at(m,i,j);
            Color c = {val*255,0,(1-val)*255,255};

            DrawRectangle(x,y,NEU_SIZE,NEU_SIZE,c);
        }
    }
}
void draw_model(NN_Model model){
    int max_cols = 0;
    for(size_t i = 0 ; i <= model.layers ; i++){
        //The core idea here is that we save the cols of the 
        //weights to know how to space the drawings
        int x = NEU_SIZE*max_cols + NEU_SIZE/2;
        int y = NEU_SIZE*i+NEU_SIZE/2;
        draw_mat(model.wi[i],x,y);
        draw_mat(model.bi[i],x,y+NEU_SIZE*model.wi[i].rows);
        max_cols = (int)model.wi[i].cols;
    }
}

#endif // !SEE_IMPLI
