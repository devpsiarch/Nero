//TODO : Mat/NN_Model check status for debuggin 
//TODO : more activation functions
//TODO : maybe its a good idea to get the ti and to in the model aswell
//TODO : model free
//the header part 
#ifndef NERO_H
#define NERO_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stddef.h>

#define Mat_at(m,r,c) (m).ptr[(r)*(m).stride+(c)]
#define Mat_free(m) free((m).ptr)
#define Mat_SHOW(m) Mat_print((m),#m,0)
#define Mat_STAT(m) Mat_print_stat((m),#m)

#define array_len(arr) sizeof((arr))/sizeof((arr[0]))

#define NN_INPUT(model) (model).ai[0]
#define NN_OUTPUT(model) (model).ai[(model).layers]


//colors for beauty 
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"


typedef struct {
	size_t rows;
	size_t cols;
	size_t stride;
	float *ptr;
	//stride later for sub matrices
}Mat;


typedef struct {
	size_t layers;
	Mat *wi;
	Mat *bi;
	Mat *ai;//the number of activation layer is layers + 1
}NN_Model;

NN_Model NN_ALLOC(size_t *network,size_t network_size);
void NN_FREE(NN_Model model);
void NN_print(NN_Model model,const char *name);
void NN_rand(NN_Model model,size_t low,size_t high);
void NN_feedforward(NN_Model model);
float NN_cost(NN_Model model,Mat ti,Mat to,size_t data_size);
void NN_finit_diff(NN_Model model,NN_Model gradient,Mat ti,Mat to,size_t data_size,float eps);
void NN_gradient_update(NN_Model model,NN_Model gradient,float rate);
void NN_check(NN_Model model,Mat ti,size_t data_size);

Mat Mat_alloc(size_t rows,size_t cols);
Mat Mat_cut(float data[],size_t rows,size_t cols,size_t stride,size_t offset);
void Mat_rand(Mat m,float max,float low);
void Mat_print(Mat m,const char *name,size_t padding);
void Mat_print_stat(Mat m,const char *name);
Mat Mat_row(Mat m,size_t row);
void Mat_copy(Mat dst,Mat src);
void Mat_dot(Mat dst,Mat a,Mat b);
void Mat_add(Mat dst,Mat a,Mat b);
void Mat_sub(Mat dst,Mat a,Mat b);
void Mat_sig(Mat dst);
void Mat_sq(Mat dst);
float Mat_sum(Mat dst);
/*=============================*/
      /*Math and rand def*/
/*=============================*/

float randf();
float sigmoidf(float x);
int randi();
void Mat_rand(Mat m,float low,float max);

#endif //NERO_H
//the C code part 
#ifndef NERO_IMPLI
#define NERO_IMPLI

NN_Model NN_ALLOC(size_t *network,size_t network_size){
	//allocate enough space for weights and biases
	assert(network_size > 0);
	NN_Model model;
	model.layers = network_size - 1;
	//allocating for weights	
	model.wi =  malloc(sizeof(*model.wi)*model.layers);
	assert(model.wi != NULL);
	//allocating for biases
	model.bi =  malloc(sizeof(*model.bi)*model.layers);
	assert(model.bi != NULL);
	//allocating for activation layer includeing the input
	model.ai =  malloc(sizeof(*model.ai)*(network_size));
	assert(model.ai != NULL);

	//{2,2,1}
	//allocating sizes of matricies
	model.ai[0] = Mat_alloc(1,network[0]);
	for(size_t i = 1 ; i < network_size ; i++){
		model.wi[i-1] = Mat_alloc(network[i-1],network[i]);
		model.bi[i-1] = Mat_alloc(1,network[i]);
		model.ai[i] = Mat_alloc(1,network[i]);
	}
	return model;
}

void NN_FREE(NN_Model model){
	for(size_t i = 0 ; i < model.layers ; i++){
		Mat_free(model.wi[i]);
		Mat_free(model.bi[i]);
		Mat_free(model.ai[i]);
	}	
}

void NN_print(NN_Model model,const char *name){
	printf(ANSI_COLOR_CYAN"Model %s = {"ANSI_COLOR_RESET"\n",name);
	char buffer[256];
	size_t buffer_size = sizeof(buffer);
	size_t i = 0;
	for(i = 0; i < model.layers ; i++){
		snprintf(buffer,buffer_size,ANSI_COLOR_RED"wi[%zu]"ANSI_COLOR_RESET,i);
		Mat_print(model.wi[i],buffer,4);
		snprintf(buffer,buffer_size,ANSI_COLOR_YELLOW"bi[%zu]"ANSI_COLOR_RESET,i);
		Mat_print(model.bi[i],buffer,4);
		if(i==0){
			snprintf(buffer,buffer_size,ANSI_COLOR_GREEN"INPUT: ai[%zu]"ANSI_COLOR_RESET,i);
			Mat_print(model.ai[i],buffer,4);
			continue;
		}
		snprintf(buffer,buffer_size,ANSI_COLOR_GREEN"activation[%zu]: ai[%zu]"ANSI_COLOR_RESET,i,i);
		Mat_print(model.ai[i],buffer,4);
	}
	snprintf(buffer,buffer_size,ANSI_COLOR_GREEN"OUTPUT: ai[%zu]"ANSI_COLOR_RESET,i);
	Mat_print(model.ai[i],buffer,4);
	printf(ANSI_COLOR_CYAN"}"ANSI_COLOR_RESET"\n");
}

void NN_rand(NN_Model model,size_t low,size_t high){
	for(size_t i = 0 ; i < model.layers ; i++){
		Mat_rand(model.wi[i],low,high);	
		Mat_rand(model.bi[i],low,high);
	}
}

void NN_feedforward(NN_Model model){
	//ai[i+1] wont overshoot cuz its bigger then the parameters with + 1
	for(size_t i = 0 ; i < model.layers ; i++){
		Mat_dot(model.ai[i+1],model.ai[i],model.wi[i]);
		Mat_add(model.ai[i+1],model.ai[i+1],model.bi[i]);
		Mat_sig(model.ai[i+1]);
	}
}

float NN_cost(NN_Model model,Mat ti,Mat to,size_t data_size){
	//assertions of sizes
	//its ok to not check rows cuz they will have the same while compairing ...
	//i think , i might have to check this if it broke 
	assert(to.cols == NN_OUTPUT(model).cols);
	assert(ti.rows == to.rows);
	Mat diff;
	float result = 0;
	//these are the input and the expacted output 
	for(size_t i = 0 ; i < data_size ; i++){
		Mat x  = Mat_row(ti,i);
		Mat y = Mat_row(to,i);

		Mat_copy(NN_INPUT(model),x);
        NN_feedforward(model);

		diff = Mat_alloc(y.rows,y.cols);
		Mat_sub(diff,y,NN_OUTPUT(model));
		Mat_sq(diff);
		result += Mat_sum(diff);	
	}
	Mat_free(diff);
	return (result/data_size);
}

void NN_finit_diff(NN_Model model,NN_Model gradient,Mat ti,Mat to,size_t data_size,float eps){
	float saved;
	float c = NN_cost(model,ti,to,data_size);
	//finite diff for all weights 
	for(size_t i = 0 ; i < model.layers ; i++){
		for(size_t j = 0 ; j < model.wi[i].rows; j++){
			for(size_t k = 0 ; k < model.wi[i].cols; k++){
				saved = Mat_at(model.wi[i],j,k);
				Mat_at(model.wi[i],j,k) += eps;
				Mat_at(gradient.wi[i],j,k) = (NN_cost(model,ti,to,data_size) - c)/eps;
				Mat_at(model.wi[i],j,k) = saved;
			}
		}	
	}
	//finite diff for all biases
	for(size_t i = 0 ; i < model.layers ; i++){
		for(size_t j = 0 ; j < model.bi[i].rows; j++){
			for(size_t k = 0 ; k < model.bi[i].cols; k++){
				saved = Mat_at(model.bi[i],j,k);
				Mat_at(model.bi[i],j,k) += eps;
				Mat_at(gradient.bi[i],j,k) = (NN_cost(model,ti,to,data_size) - c)/eps;
				Mat_at(model.bi[i],j,k) = saved;
			}
		}	
	}
}

void NN_gradient_update(NN_Model model,NN_Model gradient,float rate){
	//updating the weights and biases using gradint decent
	for(size_t i = 0 ; i < model.layers ; i++){
		for(size_t j = 0 ; j < model.wi[i].rows; j++){
			for(size_t k = 0 ; k < model.wi[i].cols; k++){
				Mat_at(model.wi[i],j,k) -= rate*Mat_at(gradient.wi[i],j,k);
			}
		}	
	}

	for(size_t i = 0 ; i < model.layers ; i++){
		for(size_t j = 0 ; j < model.bi[i].rows; j++){
			for(size_t k = 0 ; k < model.bi[i].cols; k++){
				Mat_at(model.bi[i],j,k) -= rate*Mat_at(gradient.bi[i],j,k);
			}
		}	
	}
}

void NN_check(NN_Model model,Mat ti,size_t data_size){
	assert(ti.cols == NN_INPUT(model).cols);
	printf("<%d done> -----------------------------\n",0);
	for(size_t i = 0 ; i < data_size ; i++){
		Mat x  = Mat_row(ti,i);
		Mat_copy(NN_INPUT(model),x);
        
		NN_feedforward(model);
		Mat_SHOW(NN_INPUT(model));
		Mat_SHOW(NN_OUTPUT(model));	
		printf("<%zu done> -----------------------------\n",i+1);
	}
}


Mat Mat_alloc(size_t rows,size_t cols){
	Mat matrix;
	matrix.rows = rows;
	matrix.cols = cols;
	matrix.stride = cols;
	matrix.ptr = malloc(sizeof(*matrix.ptr)*rows*cols);
	assert(matrix.ptr != NULL);
	return matrix;
}

Mat Mat_cut(float data[],size_t rows,size_t cols,size_t stride,size_t offset){
	//TODO : offset is row - 2 
	return (Mat) {
        .rows = rows,
        .cols = cols,
        .stride = stride,
        .ptr = data + offset,
    };
}

void Mat_print(Mat m,const char *name,size_t padding){
	printf("%*s ",(int)padding,"");
	printf("%s[\n",name);
	for(size_t i = 0 ; i < m.rows ; i++){
		printf("%*s",(int) padding,"");
		for(size_t j = 0 ; j < m.cols ; j++){
			printf("    %f ",Mat_at(m,i,j));
		}
		printf("\n");
	}
	printf("%*s]\n",(int)(padding+2),"");
}

void Mat_print_stat(Mat m,const char *name){
	printf("Mat : %s \n",name);
	printf("==> rows : %zu\n",m.rows);
	printf("==> cols : %zu\n",m.cols);
}

void Mat_dot(Mat dst,Mat a,Mat b){
	//assertion for correct size
	//result (a.rows , b.cols)
	assert(a.cols == b.rows);
	assert(dst.rows == a.rows);
	assert(dst.cols == b.cols);
	for(size_t i = 0 ; i < a.rows;i++){
		for(size_t j = 0 ; j < b.cols ;j++){
			Mat_at(dst,i,j) = 0;
			for(size_t k = 0 ; k < a.cols ;k++){
				Mat_at(dst,i,j) += Mat_at(a,i,k)*Mat_at(b,k,j); 
			}
		}
	}
}

Mat Mat_row(Mat m,size_t row){
	return (Mat){
	.rows = 1,
	.cols = m.cols,
	.stride = m.stride,
	.ptr = &Mat_at(m,row,0),
	};
}
void Mat_copy(Mat dst,Mat src){
	assert(dst.rows == src.rows);
	assert(dst.cols == src.cols);
	for(size_t i = 0 ;i < dst.rows ; i++){
		for(size_t j = 0 ; j < dst.cols;j++){
			Mat_at(dst,i,j) = Mat_at(src,i,j);
		}
	}	
}

void Mat_add(Mat dst,Mat a,Mat b){
	//assertion of sizes
	assert(a.cols == b.cols && a.rows == b.rows);
	assert(dst.rows == a.rows && dst.cols == a.cols);
	for(size_t i = 0 ; i < a.rows ; i++){
		for(size_t j = 0 ; j < a.cols ; j++){
			Mat_at(dst,i,j)=Mat_at(a,i,j)+Mat_at(b,i,j);
		}
	}	
}

void Mat_sub(Mat dst,Mat a,Mat b){
	//assertion of sizes
	assert(a.cols == b.cols && a.rows == b.rows);
	assert(dst.rows == a.rows && dst.cols == a.cols);
	for(size_t i = 0 ; i < a.rows ; i++){
		for(size_t j = 0 ; j < a.cols ; j++){
			Mat_at(dst,i,j)=Mat_at(a,i,j)-Mat_at(b,i,j);
		}
	}
}

void Mat_sig(Mat dst){
	for(size_t i = 0 ; i < dst.rows ; i ++){
		for(size_t j = 0 ; j < dst.cols ; j ++){
			Mat_at(dst,i,j) = sigmoidf(Mat_at(dst,i,j));
		}
	}
}

void Mat_rand(Mat m,float low,float max){
	for(size_t i = 0 ; i < m.rows ; i ++){
		for(size_t j = 0 ; j < m.cols ; j++){
			//change the values herer
			Mat_at(m,i,j) = randf()*(max - low) + low;
		}
	}
}

void Mat_sq(Mat dst){
	for(size_t i = 0 ; i < dst.rows ; i++){
		for(size_t j = 0 ; j < dst.cols ; j++){
			Mat_at(dst,i,j) = Mat_at(dst,i,j)*Mat_at(dst,i,j);
		}
	}
}

float Mat_sum(Mat dst){
	float result = 0 ; 
	for(size_t i = 0 ; i < dst.rows ; i++){
		for(size_t j = 0 ; j < dst.cols ; j++){
			result += Mat_at(dst,i,j);
		}
	}
	return result;
}

/*=============================*/
     /*Math and rand impli*/
/*=============================*/
#include <math.h>
#include <time.h>

float randf(){
	return ((float)rand() / (float)RAND_MAX);
}

int randi(){
	return rand()%11;
} 

float sigmoidf(float x){
	return 1.f / (1.f + expf(-x));
}

#endif //NERO_IMPLI
