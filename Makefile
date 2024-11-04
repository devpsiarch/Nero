file = nn

#this is the tester that i didnt break the framework ...
all:
	clang -Wall -Wextra -lm $(file).c -o $(file)
	./$(file)

#this is used to clean the binry files when am about to commit changes ... i know
clean:
	rm nn
	rm degitrec/degit
	rm gh/gh
