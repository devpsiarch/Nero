#include "mnist.h"
/*  from mnist inclution
    train image : train_image[60000][784] (type: double, normalized, flattened)
    train label : train_label[60000] (type: int)
    test image : test_image[10000][784] (type: double, normalized, flattened)
    test label : test_label[10000] (type: int)
*/

//nasty code fix later future me 

int main(void){
    srand(time(NULL));
    load_mnist();
	print_mnist_pixel(train_image,train_label,60000);
	return 0;
}
