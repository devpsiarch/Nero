#include "Nero.h"
#define NERO_IMPLI


int main(void){
	srand(time(0));
	Mat m = Mat_alloc(10,10);	
	Mat_rand(m,0,10);
	Mat_print(m);	

	Mat_free(m);
	return 0;
}
