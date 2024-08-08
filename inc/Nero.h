//the header part 
#ifndef NERO_H
#define NERO_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stddef.h>

typedef struct {
	size_t rows;
	size_t cols;
	size_t stride;
	float *ptr;
	//stride later for sub matrices
}Mat;

#define Mat_at(m,r,c) (m).ptr[(r)*(m).stride+(c)]
#define Mat_free(m) free((m).ptr)
#define Mat_SHOW(m) Mat_print((m),#m)

Mat Mat_alloc(size_t rows,size_t cols);
Mat Mat_cut(float data[],size_t rows,size_t cols,size_t stride,size_t offset);
void Mat_rand(Mat m,float max,float low);
void Mat_print(Mat m,const char *name);
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

void Mat_print(Mat m,const char *name){
	printf("%s [\n",name);
	for(size_t i = 0 ; i < m.rows ; i++){
		for(size_t j = 0 ; j < m.cols ; j++){
			printf("    %f ",Mat_at(m,i,j));
		}
		printf("\n");
	}
	printf("]\n");
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
