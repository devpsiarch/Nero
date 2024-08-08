//XOR in the framework
#include "inc/Nero.h"

typedef float sample[3];

sample or[] = {
        {0,0,0},
        {1,0,1},
        {0,1,1},
        {1,1,1},
};
sample and[] = {
        {0,0,0},
        {1,0,0},
        {0,1,0},
        {1,1,1},
};
sample nand[] = {
        {0,0,1},
        {1,0,1},
        {0,1,1},
        {1,1,0},
};
sample xor[] = {
        {0,0,0},
        {1,0,1},
        {0,1,1},
        {1,1,0},
};
sample add[] = {
        {0,0,0},
        {1,0,1},
        {0,1,1},
        {1,1,2},
};
sample *train = xor;



typedef struct {
	Mat a0; 
	Mat w1,b1,a1; 
	Mat w2,b2,a2; 
}model;

void Mat_sig(Mat m){
	for(size_t i = 0 ; i < m.rows;i++){
		for(size_t j = 0 ; j < m.cols ; j++){
			Mat_at(m,i,j) = sigmoidf(Mat_at(m,i,j));
		}
	}
}

float feedforward(model m,float x1,float x2){
	Mat_at(m.a0,0,0) = x1;
	Mat_at(m.a0,0,1) = x2;

	Mat_dot(m.a1,m.a0,m.w1);
	Mat_add(m.a1,m.a1,m.b1);
	Mat_sig(m.a1);

	Mat_dot(m.a2,m.a1,m.w2);
	Mat_add(m.a2,m.a2,m.b2);
	Mat_sig(m.a2);

	float y = *m.a2.ptr;
	return y; 
}

float cost(model m){
	size_t size = 4;
	float result = 0;
	for(size_t i = 0 ; i < size ;i++){
		float y = feedforward(m,train[i][0],train[0][1]);
		float d = train[i][2] - y;
		result += d*d;
	}
	return (result /=4);
}

void training(model *m){
	float rate = 1e-1;
	float h = 1e-1;
	float c = cost(*m);
	float tmp,diff;

	for(size_t i = 0 ; i < m->w1.rows ; i++){
		for(size_t j = 0 ; j < m->w1.cols ;j++){
			tmp = Mat_at(m->w1,i,j);
			Mat_at(m->w1,i,j) += h;
			diff = (cost(*m) - c)/h;
			Mat_at(m->w1,i,j) = tmp;
			Mat_at(m->w1,i,j) -= rate*diff;
		}
	}
	for(size_t i = 0 ; i < m->w2.rows ; i++){
		for(size_t j = 0 ; j < m->w2.cols ;j++){
			tmp = Mat_at(m->w2,i,j);
			Mat_at(m->w2,i,j) += h;
			diff = (cost(*m) - c)/h;
			Mat_at(m->w2,i,j) = tmp;
			Mat_at(m->w2,i,j) -= rate*diff;
		}
	}
	for(size_t i = 0 ; i < m->b1.rows ; i++){
		for(size_t j = 0 ; j < m->b1.cols ;j++){
			tmp = Mat_at(m->b1,i,j);
			Mat_at(m->b1,i,j) += h;
			diff = (cost(*m) - c)/h;
			Mat_at(m->b1,i,j) = tmp;
			Mat_at(m->b1,i,j) -= rate*diff;
		}
	}
	for(size_t i = 0 ; i < m->b2.rows ; i++){
		for(size_t j = 0 ; j < m->b2.cols ;j++){
			tmp = Mat_at(m->b2,i,j);
			Mat_at(m->b2,i,j) += h;
			diff = (cost(*m) - c)/h;
			Mat_at(m->b2,i,j) = tmp;
			Mat_at(m->b2,i,j) -= rate*diff;
		}
	}
}

int main(void){
	srand(time(0));
	
	model m;

	m.a0 = Mat_alloc(1,2);
	m.w1 = Mat_alloc(2,2);
	m.b1 = Mat_alloc(1,2);
	m.a1 = Mat_alloc(1,2);
	m.w2 = Mat_alloc(2,1);
	m.b2 = Mat_alloc(1,1);
	m.a2 = Mat_alloc(1,1);
	
	Mat_rand(m.w1,0,1);
	Mat_rand(m.b1,0,1);
	Mat_rand(m.w2,0,1);
	Mat_rand(m.b2,0,1);

	printf("the init model is \n");

	Mat_SHOW(m.w1);
	Mat_SHOW(m.b1);
	Mat_SHOW(m.w2);
	Mat_SHOW(m.b2);

	
	printf("after feed forward : %f\n",feedforward(m,train[0][0],train[0][1]));
	printf("init cost: %f\n",cost(m));
	for(size_t i = 0 ; i < 100000 ; i++){
		training(&m);	
	}
	printf("after training cost: %f\n",cost(m));

	return 0;
}
