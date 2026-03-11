CC      = gcc
CFLAGS  = -O2 -fPIC -Wall -Wextra -std=c11
LDFLAGS = -shared -lm

TARGET  = libsaddle.so
SRC     = surfaces.c adam.c
OBJ     = $(SRC:.c=.o)

# WASM build (requires emcc from Emscripten SDK)
EMCC     = emcc
WASM_OUT = saddle.js
WASM_DIR = ../../frontend/public/wasm

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c surfaces.h adam.h
	$(CC) $(CFLAGS) -c $< -o $@

wasm: surfaces.c adam.c wasm_api.c surfaces.h adam.h
	$(EMCC) -O2 -s WASM=1 \
	    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' \
	    -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
	    -s ALLOW_MEMORY_GROWTH=1 \
	    -s MODULARIZE=1 \
	    -s EXPORT_NAME="SaddleModule" \
	    -s ENVIRONMENT=web \
	    surfaces.c adam.c wasm_api.c \
	    -o $(WASM_OUT) -lm

install-wasm: wasm
	mkdir -p $(WASM_DIR)
	cp saddle.js saddle.wasm $(WASM_DIR)/

clean:
	rm -f $(OBJ) $(TARGET) libsurfaces.so $(WASM_OUT) saddle.wasm

.PHONY: all clean wasm install-wasm
