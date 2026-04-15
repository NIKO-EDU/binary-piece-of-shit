CC      = gcc
SRCDIR  = src
CFLAGS_SO   = -O3 -march=native -fopenmp -shared -fPIC -lm
CFLAGS_TEST = -O1 -g -fsanitize=address,undefined -Wall -Wextra -lm

.PHONY: all test clean

all: $(SRCDIR)/kernel.so

$(SRCDIR)/kernel.so: $(SRCDIR)/kernel.c
	$(CC) $(CFLAGS_SO) -o $@ $<

$(SRCDIR)/kernel_test: $(SRCDIR)/kernel.c
	$(CC) $(CFLAGS_TEST) -DKERNEL_TEST_MAIN -o $@ $<

test: $(SRCDIR)/kernel_test
	./$(SRCDIR)/kernel_test

clean:
	rm -f $(SRCDIR)/kernel.so $(SRCDIR)/kernel_test
