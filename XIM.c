//XOR in the framework
#include "inc/Nero.h"

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

void feedforward(model m){
	Mat_dot(m.a1,m.a0,m.w1);
	Mat_add(m.a1,m.a1,m.b1);
	Mat_sig(m.a1);

	Mat_dot(m.a2,m.a1,m.w2);
	Mat_add(m.a2,m.a2,m.b2);
	Mat_sig(m.a2);
}

float cost(model m,Mat ti,Mat to){
	assert(ti.rows == to.rows);
	assert(to.cols == m.a2.cols);

	size_t n = ti.rows;
	float c = 0;

	for(size_t i = 0 ; i < n ; i++){
		Mat x = Mat_row(ti,i);
		Mat y = Mat_row(to,i);

		Mat_copy(m.a0,x);
		feedforward(m);

		size_t l = to.cols;
		for(size_t j = 0 ; j < l ; j++){
			float d = Mat_at(m.a2,0,j) - Mat_at(y,0,j);
			c += d*d;
		}
	}
	return c/n;
}

void finit_diff(model m,model grad,Mat ti,Mat to){
	float c = cost(m,ti,to);
	float eps = 1e-1;
	float saved;

	for(size_t i = 0 ; i < m.w1.rows ; i++){
		for(size_t j = 0 ; j < m.w1.cols ; j++){
            saved = Mat_at(m.w1,i,j);
            Mat_at(m.w1,i,j) += eps;
            Mat_at(grad.w1,i,j) = (cost(m,ti,to) - c)/eps;
            Mat_at(m.w1,i,j) = saved;
		}
	}
	for(size_t i = 0 ; i < m.w2.rows ; i++){
		for(size_t j = 0 ; j < m.w2.cols ; j++){
			saved = Mat_at(m.w2,i,j);
			Mat_at(m.w2,i,j) += eps;
			Mat_at(grad.w2,i,j) = (cost(m,ti,to) - c)/eps;
			Mat_at(m.w2,i,j) = saved ;
		}
	}
	for(size_t i = 0 ; i < m.b1.rows ; i++){
		for(size_t j = 0 ; j < m.b1.cols ; j++){
			saved = Mat_at(m.b1,i,j);
			Mat_at(m.b1,i,j) += eps;
			Mat_at(grad.b1,i,j) = (cost(m,ti,to) - c)/eps;
			Mat_at(m.b1,i,j) = saved ;
		}
	}
	for(size_t i = 0 ; i < m.b2.rows ; i++){
		for(size_t j = 0 ; j < m.b2.cols ; j++){
			saved = Mat_at(m.b2,i,j);
			Mat_at(m.b2,i,j) += eps;
			Mat_at(grad.b2,i,j) = (cost(m,ti,to) - c)/eps;
			Mat_at(m.b2,i,j) = saved ;
		}
	}
}

void update(model m,model grad){
	float rate = 1e-1;
	for(size_t i = 0 ; i < m.w1.rows ; i++){
		for(size_t j = 0 ; j < m.w1.cols ; j++){
			Mat_at(m.w1,i,j) -= rate*Mat_at(grad.w1,i,j);
		}
	}
	for(size_t i = 0 ; i < m.w2.rows ; i++){
		for(size_t j = 0 ; j < m.w2.cols ; j++){
			Mat_at(m.w2,i,j) -= rate*Mat_at(grad.w2,i,j);
		}
	}
	for(size_t i = 0 ; i < m.b1.rows ; i++){
		for(size_t j = 0 ; j < m.b1.cols ; j++){
			Mat_at(m.b1,i,j) -= rate*Mat_at(grad.b1,i,j);
		}
	}
	for(size_t i = 0 ; i < m.b2.rows ; i++){
		for(size_t j = 0 ; j < m.b2.cols ; j++){
			Mat_at(m.b2,i,j) -= rate*Mat_at(grad.b2,i,j);
		}
	}
}

float train[] = {
	0, 0, 0,
	1, 0, 1,
	0, 1, 1,
	1, 1, 0,
};


int main(void){
	srand(time(0));
	size_t stride = 3;
	size_t n = sizeof(train)/sizeof(train[0])/stride;

	Mat ti = Mat_cut(train,n,2,3,0);	
	Mat to = Mat_cut(train,n,1,3,2);	


	model m;
	model grad;

	m.a0 = Mat_alloc(1,2);
	m.w1 = Mat_alloc(2,2);
	m.b1 = Mat_alloc(1,2);
	m.a1 = Mat_alloc(1,2);
	m.w2 = Mat_alloc(2,1);
	m.b2 = Mat_alloc(1,1);
	m.a2 = Mat_alloc(1,1);


	grad.a0 = Mat_alloc(1,2);
	grad.w1 = Mat_alloc(2,2);
	grad.b1 = Mat_alloc(1,2);
	grad.a1 = Mat_alloc(1,2);
	grad.w2 = Mat_alloc(2,1);
	grad.b2 = Mat_alloc(1,1);
	grad.a2 = Mat_alloc(1,1);


	Mat_rand(m.w1,0,1);
	Mat_rand(m.b1,0,1);
	Mat_rand(m.w2,0,1);
	Mat_rand(m.b2,0,1);

	printf("the init model is \n");

	Mat_SHOW(m.w1);
	Mat_SHOW(m.b1);
	Mat_SHOW(m.w2);
	Mat_SHOW(m.b2);

	
	Mat_SHOW(ti);
	Mat_SHOW(to);


	printf("the cost of the model %f\n",cost(m,ti,to));

	for(size_t i = 0 ; i < 200000 ; i++){
		finit_diff(m,grad,ti,to);
		update(m,grad);
        printf("cost is : %f\n",cost(m,ti,to));
	}
	//Mat_SHOW(grad.w1);
	//Mat_SHOW(grad.b1);
	//Mat_SHOW(grad.w2);
	//Mat_SHOW(grad.b2);
	return 0;
}
