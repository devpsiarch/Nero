#include "../inc/Nero.h"
#include "../inc/read.h"



int main(){
    int lines_read;
    char *cat_attributes[] = {"g","h"};
    float replacement_values[] = {1.f,0.f};

    float* data = read_csv("gh/data/train.data"
                               ,11
                               ,&lines_read
                               ,2
                               ,cat_attributes
                               ,replacement_values);

     
    srand(69);
    size_t arch[] = {10,5,1};
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);

    size_t stride = 11;
	  size_t n = lines_read;

    Mat ti = Mat_cut(data,n,arch[0],stride,0);
	  Mat to = Mat_cut(data,n,last_element(arch),stride,to_get_offset(stride,last_element(arch)));	

    Mat_STAT(ti);
    Mat_STAT(to);

        for(size_t i = 0 ; i < 10 ; i++){
            NN_backprop(model,gradient,ti,to);
            NN_gradient_update(model,gradient,0.5);
            printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,n));
        }

defer_model:
    NN_FREE(model);
    NN_FREE(gradient);
defer_main:
    free(data);
    return 0;
}
