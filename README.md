# Nero: Neural network header only library 
![vis](./resources/vis.png)

Nero is a header only library that provides machine learning algorithms written in pure C.

---

# How to use 
You can follow the bellow template for starting the learning procces : 

```c
#include "../inc/Nero.h"
#include "../inc/read.h"
/* Example on gamma/Hadron dataset */

int main(){
    // specify csv file and replacements for non-numerical values
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
    // define the form/architecture of the NN (layers and all)
    size_t arch[] = {10,5,1};
    NN_Model model    = NN_ALLOC(arch,array_len(arch));
    NN_Model gradient = NN_ALLOC(arch,array_len(arch));
    NN_rand(model,0,1);

    size_t stride = 11;
	size_t n = lines_read;

    // cut a portion of the raw data to a matrix representation
    Mat ti = Mat_cut(data,n,arch[0],stride,0);
    Mat to = Mat_cut(data,n,last_element(arch),stride,to_get_offset(stride,last_element(arch)));	

    // Showing info abt matrix (row*col)
    Mat_STAT(ti);
    Mat_STAT(to);

        // learning proccess
        for(size_t i = 0 ; i < 1000 ; i++){
            NN_backprop(model,gradient,ti,to);
            NN_gradient_update(model,gradient,0.5);
            printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,n));
        }

// free model and IO matrices
defer_model:
    NN_FREE(model);
    NN_FREE(gradient);
    Mat_free(ti);
    Mat_free(to);
defer_main:
    free(data);
    return 0;
}
```
---
# Limitations 
For now the Nero.h provides only the `sigmoid` activation function and its respected `backpopagation` algorithm.
More will be added soon.

--- 
# Dependencies 
We depend only on `libc` and `raylib` for visualization , run `get_deps.sh` on Linux.

---

# Goal
The goal from this framework is to understand how neural network and AI work by writting your an implimentation from scratch and tinker around trying to wrape your mind around how big frameworks are made , so that when you are using them you get an deeper understanding of how they work under the hood .

---
# Future Goals for Nero ? 
- More flexibility and options
- Getting a Hardware boost (Using GPUs).
- Running a language/vision model on it.
