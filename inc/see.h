#ifndef SEE_H
#define SEE_H
/*=============================*/
    /*visiulizing the model*/
/*=============================*/
#include "Nero.h"
#include <raylib.h>
//i dont wanna pass around these values everywhere
//maybe when i make a better API
// the values here are only used by the cube displayer
#define WIDTH 1000
#define HIGHT 700
#define NEU_SIZE 50

#define PICK_COLOR(value) {100,(1-(val))*255,(val)*255,255}

/* Intermediate representation for the location of nodes */
typedef struct {
    size_t size;
    float*dx;
    float**dy;
}NN_map;

NN_map* init_nnmap(size_t arch[],size_t size,int w,int h);
void kill_nnmap(NN_map* m);

/* draws the familiar nerual network with nodes and weights connecting them */ 
void draw_nn(NN_Model m,NN_map* map,size_t arch[],size_t size);

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

NN_map* init_nnmap(size_t arch[],size_t size,int w,int h){
    float dx = w / size;
    NN_map* map = malloc(sizeof(NN_map));
    map->size = size;
    map->dx = malloc(sizeof(float)*size);
    map->dy = malloc(sizeof(float*)*size);
    for(size_t i = 0 ; i < size ; i++){
        map->dx[i] = (i+1)*dx-dx/2;
        map->dy[i] = malloc(sizeof(float)*arch[i]);
        for(size_t j = 0 ; j < arch[i] ; ++j){
            float dy = h / arch[i];
            map->dy[i][j] = (j+1)*dy-dy/2;
        }
    }

    return map;
}

void kill_nnmap(NN_map* m){
    free(m->dx);
    for(size_t i = 0 ; i < m->size ; i++){
        free(m->dy[i]);
    }
    free(m->dy);
    free(m);
}

void draw_nn(NN_Model m,NN_map* map,size_t arch[],size_t size){
    for(size_t i = 0 ; i < size ; i++){
        for(size_t j = 0 ; j < arch[i] ; j++){
            if(i == 0){
                DrawCircle(map->dx[i],map->dy[i][j],10,GRAY); 
                for(size_t k = 0 ; k < arch[i+1] ; k ++){
                    float val = Mat_at(m.wi[i],j,k);
                    Color c = PICK_COLOR(val);
                    DrawLine(map->dx[i],map->dy[i][j],map->dx[i+1],map->dy[i+1][k],c);
                }
            }else{
                float val = Mat_at(m.bi[i-1],0,j);
                Color c = PICK_COLOR(val);
                DrawCircle(map->dx[i],map->dy[i][j],10,c); 
                if(i != size-1){
                    for(size_t k = 0 ; k < arch[i+1] ; k ++){
                        float val = Mat_at(m.wi[i],j,k);
                        Color c = PICK_COLOR(val);
                        DrawLine(map->dx[i],map->dy[i][j],map->dx[i+1],map->dy[i+1][k],c);
                    }
                }
            }
        }
    }
}

#endif // !SEE_IMPLI
