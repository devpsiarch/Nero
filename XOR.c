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

//define a type to ease life dude
typedef float sample[3]; 

typedef struct {
	float per1_w1;
	float per1_w2;
	float per1_b;

	float per2_w1;
	float per2_w2;
	float per2_b;

	float per3_w1;
	float per3_w2;
	float per3_b;
}model;

model get_model(){
	model m;
	m.per1_w1 = randf();
	m.per1_w2 = randf();
	m.per1_b = randf();

	m.per2_w1 = randf();
	m.per2_w2 = randf();
	m.per2_b = randf();

	m.per3_w1 = randf();
	m.per3_w2 = randf();
	m.per3_b = randf();
	return m;
}

void print_model(model m){
	printf("per1_w1 = %f \n",m.per1_w1);
	printf("per1_w2 = %f \n",m.per1_w2);
	printf("per1_b = %f \n",m.per1_b);

	printf("per2_w1 = %f \n",m.per2_w1);
	printf("per2_w2 = %f \n",m.per2_w2);
	printf("per2_b = %f \n",m.per2_b);

	printf("per3_w1 = %f \n",m.per3_w1);
	printf("per3_w2 = %f \n",m.per3_w2);
	printf("per3_b = %f \n",m.per3_b);
}

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

sample *train = xor;

float forward(model m,float x1,float x2){
	float val1 = sigmoidf(x1*m.per1_w1 + x2*m.per1_w2 + m.per1_b);
	float val2 = sigmoidf(x1*m.per2_w1 + x2*m.per2_w2 + m.per2_b);
	float y = sigmoidf(val1*m.per3_w1 + val2*m.per3_w2 + m.per3_b);
	return y;
}

float cost(model m){
	float result = 0;
	for(int i = 0 ; i < 4;i++){
		float y = forward(m,train[i][0],train[i][1]);
		float d = train[i][2] - y;
		result += d*d;
	}
	result /= 4;
	return result;
}

void training(model *m){
	//derivatives w/ respect to each parameter
	float ncost = cost(*m);
	float rate = 1e-2;
	float h = 1e-2;
	//stores the resent derivation respect
	float temp,dp;
	
	temp = m->per1_w1;
	m->per1_w1 += h;
	dp = (cost(*m) - ncost)/h;
	m->per1_w1 = temp;
	m->per1_w1 -= rate*dp;

	temp = m->per1_w2;
	m->per1_w2 += h;
	dp = (cost(*m) - ncost)/h;
	m->per1_w2 = temp;
	m->per1_w2 -= rate*dp;

	temp = m->per1_b;
	m->per1_b += h;
	dp = (cost(*m) - ncost)/h;
	m->per1_b = temp;
	m->per1_b -= rate*dp;



	temp = m->per2_w1;
	m->per2_w1 += h;
	dp = (cost(*m) - ncost)/h;
	m->per2_w1 = temp;
	m->per2_w1 -= rate*dp;

	temp = m->per2_w2;
	m->per2_w2 += h;
	dp = (cost(*m) - ncost)/h;
	m->per2_w2 = temp;
	m->per2_w2 -= rate*dp;

	temp = m->per2_b;
	m->per2_b += h;
	dp = (cost(*m) - ncost)/h;
	m->per2_b = temp;
	m->per2_b -= rate*dp;



	temp = m->per3_w1;
	m->per3_w1 += h;
	dp = (cost(*m) - ncost)/h;
	m->per3_w1 = temp;
	m->per3_w1 -= rate*dp;

	temp = m->per3_w2;
	m->per3_w2 += h;
	dp = (cost(*m) - ncost)/h;
	m->per3_w2 = temp;
	m->per3_w2 -= rate*dp;

	temp = m->per3_b;
	m->per3_b += h;
	dp = (cost(*m) - ncost)/h;
	m->per3_b = temp;
	m->per3_b -= rate*dp;

}

void check(model m){
	for(int i = 0 ;i < 4;i++){
		printf("%f ope %f => %f\n",
				train[i][0],
				train[i][1],
				forward(m,train[i][0],train[i][1]));
	}
}

int main(void){
	model m = get_model();
	print_model(m);
	for(int i = 0 ; i < 1000000 ; i++){
		training(&m);
		//printf("cst : %f\n",cost(m));
	}
	printf("cst : %f\n",cost(m));

	check(m); 	
	return 0;
}
