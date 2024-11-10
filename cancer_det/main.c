#include "../inc/Nero.h"
float data[569][31];
int extract(char *path){
    // data has : "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst".
    int read = 0;
    int record = 0;
    char gh;
    int id;
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
            "%d,%c,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
            &id,
            &gh, 
            &data[record][0],
            &data[record][1],
            &data[record][2],
            &data[record][3],
            &data[record][4],
            &data[record][5],
            &data[record][6],
            &data[record][7],
            &data[record][8],
            &data[record][9],
            &data[record][10],
            &data[record][11],
            &data[record][12],
            &data[record][13],
            &data[record][14],
            &data[record][15],
            &data[record][16],
            &data[record][17],
            &data[record][18],
            &data[record][19],
            &data[record][20],
            &data[record][21],
            &data[record][22],
            &data[record][23],
            &data[record][24],
            &data[record][25],
            &data[record][26],
            &data[record][27],
            &data[record][28],
            &data[record][29]);
        //printf("%d\n",read);
        if (gh =='B') data[record][30] = 1.f; // turns g to 1 and h to 0 
        if (gh == 'M') data[record][30] = 0.f;
        if (read == 32) record ++;
        else if (read != 32 && !feof(file)){
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
void print_data(void){
    for(size_t i = 0 ; i < 569 ; i++){
        for(size_t j = 0 ; j < 31 ; j++){
            printf("%f ",data[i][j]);
        }
        printf("\n");
    }
}
int main(void){

    extract("data/data.csv");
    srand(69);
    size_t arch[] = {30,10,10,1};
    NN_Model model = NN_ALLOC(arch,array_len(arch)); 
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);

    size_t stride = 31;
    size_t rows = 569;
    Mat ti = Mat_cut(data,rows,arch[0],stride,0);
    Mat to = Mat_cut(data,rows,last_element(arch),stride,to_get_offset(stride,last_element(arch)));

    printf("%f \n",NN_cost(model,ti,to,rows));
    for(size_t i = 0 ; i < 1000 ; i++){
        //NN_finit_diff(model,gradient,ti,to,n,1);
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,1);
        printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,rows));
    }  

    return 0;
}
