file = degitrec/degit.c
bin = bin/degit

build:
	gcc -O3 -fsanitize=address -g -Wall -Wextra -lm -lraylib $(file) -o $(bin)
run:build
	./$(bin)
clean:
	rm bin/*
