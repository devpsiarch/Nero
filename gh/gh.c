#include "../inc/Nero.h"
float train_dataset[18889][11];
int extract(char *path){
    int read = 0;
    int record = 0;
    char gh;
    FILE *file;
    file = fopen(path, "r");
    if (file == NULL)
    {
        printf("error reading file \n");
        return 1;
    }
    do
    { 
        read = fscanf(file,
            "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%c\n",
            &train_dataset[record][0],
            &train_dataset[record][1],
            &train_dataset[record][2],
            &train_dataset[record][3],
            &train_dataset[record][4],
            &train_dataset[record][5],
            &train_dataset[record][6],
            &train_dataset[record][7],
            &train_dataset[record][8],
            &train_dataset[record][9],
            &gh);
        if (gh =='g') train_dataset[record][10] = 1.f; // turns g to 1 and h to 0 
        if (gh == 'h') train_dataset[record][10] = 0.f;
        if (read == 11) record ++;
        if (read != 11 && !feof(file)){
            printf("file format incorrect\n");
            return 1;
        }  
        if (ferror(file)){
            printf("errore reading file ! \n"); 
            return 1;
        }
    }while (!feof(file));
    fclose(file);
    printf("<<%d>> records read \n\n",record);
    return  0;
}
void printdataset(void){
    for (int i = 0; i < 18889 ; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            printf("%12f ",train_dataset[i][j]);
        }
        printf("\n");
    }
}

int main(){
    extract("data/train.data");

    srand(69);
    size_t arch[] = {10,4,2,2,1};

    NN_Model model = NN_ALLOC(arch,array_len(arch)); 
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);

    size_t stride = 11;
    size_t rows = 18889;

    Mat ti = Mat_cut(train_dataset,rows,arch[0],stride,0);
    Mat to = Mat_cut(train_dataset,rows,last_element(arch),stride,to_get_offset(stride,last_element(arch)));	


    printf("%f \n",NN_cost(model,ti,to,rows));
    for(size_t i = 0 ; i < 10000 ; i++){
        //NN_finit_diff(model,gradient,ti,to,n,1);
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,1);
        printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,rows));
    }
    return 0;
}
