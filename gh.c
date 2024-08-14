#include "inc/Nero.h"
#include <string.h>

float train[19020*11];

void read_data(const char *filepath){
	FILE *file = fopen(filepath,"r");
	assert(file != NULL);
	
	//getting the size of file
    fseek(file, 0L, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0L, SEEK_SET);

	printf("the size of the file is %ld\n",fileSize);

	//alllocate for buffer
    char* buffer = (char*)malloc(fileSize + 1); // Allocate memory for file content
	assert(buffer != NULL);

	//gets the file in a buffer
    fread(buffer, 1, fileSize, file); // Read file content into buffer
    buffer[fileSize] = '\0'; // Null-terminate the buffer

    fclose(file);

	//printf("data is :\n%s",buffer);
	size_t index = 0;
	char *line = strtok(buffer,"\n");
	while(line != NULL){
		char c;
		sscanf(line,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%c",
				&train[index],
				&train[index+1],
				&train[index+2],
				&train[index+3],
				&train[index+4],
				&train[index+5],
				&train[index+6],
				&train[index+7],
				&train[index+8],
				&train[index+9],
				&c);
		if(c == 'h'){
			train[index+10] = 1;
		}	
		else{
			train[index+10] = 0;
		}
		index += 11;	
		line = strtok(NULL,"\n");
	}

}

void printdata(size_t stride){
	for(size_t r = 0 ; r < 19020;r++){
		for(size_t i = 0 ; i < stride ; i++){
			printf("%f ",train[r*stride+i]);
		}
		printf("\n");
	}
}

int main(void){
	read_data("gh.data");	
	
	//define the arch 
	size_t arch[] = {10,3,2,1};
	NN_Model model = NN_ALLOC(arch,array_len(arch));
	NN_Model gradient = NN_ALLOC(arch,array_len(arch));
	NN_rand(model,0,1);

	size_t stride = 11;
	size_t n = train_size(train,stride);

	Mat ti = {
		.rows = 19020,
		.cols = 10,
		.stride = 11,
		.ptr = train, 
	};
	Mat to = {
		.rows = 19020,
		.cols = 1,
		.stride = 11,
		.ptr = train+10, 
	};


    //Mat_SHOW(ti);
    //Mat_SHOW(to);
	Mat row = Mat_row(ti,0);

	Mat_STAT(NN_INPUT(model));	
	Mat_STAT(row);

    printf("init cost :%f\n",NN_cost(model,ti,to,n));
	for(size_t i = 0 ; i < 10*1000; i++){
        //NN_finit_diff(model,gradient,ti,to,n,1);
        NN_backprop(model,gradient,ti,to);
        NN_gradient_update(model,gradient,1);
        printf("%zu | cost : %f \n",i,NN_cost(model,ti,to,n));
    }
    
	//NN_check(model,ti,n);

    NN_FREE(model);
    NN_FREE(gradient);

	return 0;
}
