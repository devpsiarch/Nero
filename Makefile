file = main

build:
	gcc -O3 -fsanitize=address -g -Wall -Wextra -lm -lraylib $(file).c -o bin/$(file)
run:build
	./bin/$(file)
clean:
	rm bin/*
