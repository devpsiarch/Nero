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
void NN_print(NN_Model model,const char *name);
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
		model.ai[i-1] = Mat_alloc(1,network[i]);
	}
	return model;
}

void NN_print(NN_Model model,const char *name){
	printf(ANSI_COLOR_CYAN"Model %s = {"ANSI_COLOR_RESET"\n",name);
	char buffer[256];
	for(size_t i = 0; i < model.layers ; i++){
		snprintf(buffer,sizeof(buffer),ANSI_COLOR_RED"wi[%zu]"ANSI_COLOR_RESET,i);
		Mat_print(model.wi[i],buffer,6);
		snprintf(buffer,sizeof(buffer),ANSI_COLOR_YELLOW"bi[%zu]"ANSI_COLOR_RESET,i);
		Mat_print(model.bi[i],buffer,6);
		snprintf(buffer,sizeof(buffer),ANSI_COLOR_GREEN"ai[%zu]"ANSI_COLOR_RESET,i);
		Mat_print(model.ai[i],buffer,6);
	}
	printf(ANSI_COLOR_CYAN"}"ANSI_COLOR_RESET"\n");
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
		for(size_t j = 0 ; j < m.cols ; j++){
			printf("    %*f ",(int)(padding+6),Mat_at(m,i,j));
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
	assert(a.cols == b.rows);
	assert(dst.rows == a.rows && dst.cols == b.cols);
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
	
void Mat_rand(Mat m,float low,float max){
	for(size_t i = 0 ; i < m.rows ; i ++){
		for(size_t j = 0 ; j < m.cols ; j++){
			//change the values herer
			Mat_at(m,i,j) = randf()*(max - low) + low;
		}
	}
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
