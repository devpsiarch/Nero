#ifndef READ_H
#define READ_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE 65536
#define MALLOC(size) malloc((size))
#define CLEAN_2D_DATA(arr,lines)        \
do {                                    \
    for(int i = 0 ; i < (lines) ; i++){ \
        free((arr)[i]);                 \
    }                                   \
    free((arr));                        \
} while (false);                        \

// countes the lines in any file
int count_lines(FILE *filestream);

// reads a .csv file returns a float 2D array containing the data 
// if supplied with cat_attri and replace with the same size n 
// it shall replace any non numerical attribute with the corresponding "assumed"
// numerical value
// keep as default NULL value
float* read_csv(const char* path,size_t num_features,int* returned_lines_read,
                 int n,
                 char* cat_attri[],
                float replace[]);

#endif // !READ_H
#ifndef READ_IMPLI 
#define READ_IMPLI

int count_lines(FILE* filestream){
    char buf[BUF_SIZE];
    int count = 0;
    while(true){
        int res = fread(buf,1,BUF_SIZE,filestream);
        if(ferror(filestream)){
            printf("INFO: Error while counting lines.\n");
            goto defer;
        }
        int i;
        for(i = 0 ; i < res ; ++i){
            if(buf[i] == '\n') count++;
        }
        if(feof(filestream)) break;
    }
    return count;
defer:
    return -1;
}

float* read_csv(const char* path,size_t num_features,int* returned_lines_read,
                 int n,
                 char* cat_attri[],
                float replace[]){
    FILE* fp = fopen(path,"rb");
    if(fp == NULL){
        printf("INFO: unable to open file.\n");
        goto defer;
    }
    int lines_of_files = count_lines(fp);
    if(lines_of_files == -1){
        printf("INFO: unable to count lines of file.\n");
        goto defer;
    }
    // store the data in a contigous space
    float* data = (float*)MALLOC(lines_of_files*num_features*sizeof(float));
    if(data == NULL){
        printf("INFO: unable to allocate memory for data.\n");
        free(data);
        goto defer;       
    }
    // reset the cursor of the file pointer since it was used in count_lines
    fseek(fp,0,SEEK_SET);
    char buffer[BUF_SIZE];
    int read_lines = 0;
    while(fgets(buffer,BUF_SIZE,fp) != NULL){
        size_t features_read = 0;
        char* token;
        token = strtok(buffer,",");
        while(token != NULL){
            if(features_read > num_features){
                free(data);
                goto defer;
            }
            bool replaceflag = false;
            for(int i = 0 ; i < n ; ++i){
                // if we have a match
                if(strncmp(token,cat_attri[i],strlen(cat_attri[i])) == 0){
                    data[read_lines*num_features+features_read] = replace[i];
                    replaceflag = true;
                    break;
                }
            } 
            if(replaceflag == false) data[read_lines*num_features+features_read] = atof(token);
            //printf("read value is %f\n",data[read_lines][features_read]);
            token = strtok(NULL,",");
            ++features_read;
        }
        ++read_lines;
    }
    printf("INFO: Reading over %d/%d line read.\n",read_lines,lines_of_files);
    *returned_lines_read = read_lines;
    return data;
defer:
    if(fp != NULL) fclose(fp);
    return NULL;
}

#endif // !READ_H
