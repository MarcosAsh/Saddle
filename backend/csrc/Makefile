CC      = gcc
CFLAGS  = -O2 -fPIC -Wall -Wextra -std=c11
LDFLAGS = -shared -lm

TARGET  = libsaddle.so
SRC     = surfaces.c adam.c
OBJ     = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c surfaces.h adam.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) libsurfaces.so

.PHONY: all clean
