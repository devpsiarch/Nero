#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float sigmoidf(float x){
	return 1.f / (1.f + expf(-x));
}


float randf(){
	return ((float)rand() / (float)RAND_MAX); 
}

float or_gate[4][3] = {
	{0,0,0},
	{1,0,1},
	{0,1,1},
	{1,1,1},
};

float cost(float w1,float w2,float b){
	float result = 0;
	for(int i = 0 ; i < 4 ; i++){
		float y = or_gate[i][0]*w1 + or_gate[i][1]*w2 + b;
		y = sigmoidf(y);
		float d = or_gate[i][2] - y;
		result += d*d;
	}
	//deviding by how many data set are for training
	return result/4;
}

void train(float *w1,float *w2,float *b){
	//dcost/d(model) ==> dc/dw1 && dc/dw2 && dc/db	
	float ncost = cost(*w1,*w2,*b);

	float h = 1e-3;
	float rate = 1e-3;

	float dw1 = (cost(*w1+h,*w2,*b) - ncost)/h;
	float dw2 = (cost(*w1,*w2+h,*b) - ncost)/h;
	float db = (cost(*w1,*w2,*b+h) - ncost)/h;

	//update the parameters 
	*w1 -= rate*dw1;
	*w2 -= rate*dw2;
	*b -=  rate*db;
}

int main(void){
	//definition of the model
	srand(10);
	float w1 = randf();
	float w2 = randf();
	float b = randf();

	for(int i = 0 ; i < 4 ; i++){
		float y = or_gate[i][0]*w1 + or_gate[i][1]*w2 + b;
		printf("model : %f | expected : %f\n",y,or_gate[i][2]);
	}
	printf("at the start ==> the cost of this model is : %f\n",cost(w1,w2,b));
	
	for(int i = 0 ; i < 10000000 ; i++){
		train(&w1,&w2,&b);
		//printf("training ... cost : %f\n",cost(w1,w2,b));

	}
	printf("after train ==> the cost of this model is : %f\n",cost(w1,w2,b));

	for(int i = 0 ;i < 4;i++){
		printf("%f || %f => %f\n",or_gate[i][0],or_gate[i][1],sigmoidf(or_gate[i][0]*w1+or_gate[i][2]*w2+b));
	}

	return 0;
}
