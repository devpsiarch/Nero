file = nn

all:
	clang -Wall -Wextra -lm $(file).c -o $(file)
	./$(file)
