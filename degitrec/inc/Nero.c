#include "Nero.h"
#define NERO_IMPLI


int main(void){
	size_t arch[] = {2,2,1};
	NN_Model model = NN_ALLOC(arch,array_len(arch)); 
	NN_rand(model,0,1);
	NN_print(model,"test");
	
	return 0;
}
