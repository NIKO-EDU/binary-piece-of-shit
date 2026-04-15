CC      = gcc
SRCDIR  = src

ifeq ($(OS),Windows_NT)
    EXT         = .dll
    CFLAGS_SO   = -O3 -march=native -fopenmp -shared -lm
    TEST_BIN    = $(SRCDIR)/kernel_test.exe
    CFLAGS_TEST = -O1 -g -Wall -Wextra -lm
else
    EXT         = .so
    CFLAGS_SO   = -O3 -march=native -fopenmp -shared -fPIC -lm
    TEST_BIN    = $(SRCDIR)/kernel_test
    CFLAGS_TEST = -O1 -g -fsanitize=address,undefined -Wall -Wextra -lm
endif

TARGET = $(SRCDIR)/kernel$(EXT)

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRCDIR)/kernel.c
	$(CC) $(CFLAGS_SO) -o $@ $<

$(TEST_BIN): $(SRCDIR)/kernel.c
	$(CC) $(CFLAGS_TEST) -DKERNEL_TEST_MAIN -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(SRCDIR)/kernel.so $(SRCDIR)/kernel.dll \
	      $(SRCDIR)/kernel_test $(SRCDIR)/kernel_test.exe
