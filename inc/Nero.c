#include "Nero.h"
#define NERO_IMPLI


int main(void){
	size_t arch[] = {2,2,1};
	NN_Model model = NN_ALLOC(arch,array_len(arch)); 
	NN_print(model,"test");
	
	/*Mat_STAT(model.ai[0]);	
	Mat_STAT(model.wi[0]);
	Mat_STAT(model.bi[0]);

	Mat_STAT(model.ai[1]);
	Mat_STAT(model.wi[1]);
	Mat_STAT(model.bi[1]);*/
	return 0;
}
