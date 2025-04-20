# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wformat -I$(SRC_DIR)
LDFLAGS =

# Optimization and debug flags
ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

# Source and build directories
SRC_DIR = src
BUILD_DIR = build

# Explicitly list source files
SOURCES = $(SRC_DIR)/app.cpp $(SRC_DIR)/dllama.cpp $(SRC_DIR)/dllama-api.cpp $(SRC_DIR)/llm.cpp $(SRC_DIR)/tokenizer.cpp \
          $(SRC_DIR)/nn/nn-core.cpp $(SRC_DIR)/nn/nn-quants.cpp $(SRC_DIR)/nn/nn-executor.cpp $(SRC_DIR)/nn/nn-network.cpp \
          $(SRC_DIR)/nn/llamafile/sgemm.cpp $(SRC_DIR)/nn/nn-cpu-ops.cpp $(SRC_DIR)/nn/nn-cpu.cpp $(SRC_DIR)/nn/nn-vulkan.cpp
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS = $(OBJECTS:.o=.d)
EXECUTABLE = dllama
API_EXECUTABLE = dllama-api

# Device check
DEVICE_CHECK_SRC = device-check.cpp
DEVICE_CHECK_EXEC = device-check

# Test sources
TEST_SOURCES = $(SRC_DIR)/nn/nn-cpu-test.cpp $(SRC_DIR)/nn/nn-cpu-ops-test.cpp $(SRC_DIR)/nn/nn-vulkan-test.cpp $(SRC_DIR)/tokenizer-test.cpp
TEST_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TEST_SOURCES))
TEST_EXECUTABLES = nn-cpu-test nn-cpu-ops-test nn-vulkan-test tokenizer-test

# Vulkan support
ifdef DLLAMA_VULKAN
	CGLSLC = glslc
	CXXFLAGS += -DDLLAMA_VULKAN
	ifeq ($(OS),Windows_NT)
		LDFLAGS += -L$(VK_SDK_PATH)\lib -lvulkan-1
		CXXFLAGS += -I$(VK_SDK_PATH)\include
	else
		LDFLAGS += -lvulkan
	endif

	VULKAN_SHADER_DIR = $(SRC_DIR)/nn/vulkan
	VULKAN_SHADER_SRCS := $(wildcard $(VULKAN_SHADER_DIR)/*.comp)
	VULKAN_SHADER_BINS := $(patsubst %.comp,$(BUILD_DIR)/nn/vulkan/%.spv,$(notdir $(VULKAN_SHADER_SRCS)))
	DEPS += $(VULKAN_SHADER_BINS)
endif

# Platform-specific settings
ifeq ($(OS),Windows_NT)
	LDFLAGS += -lws2_32
	DELETE_CMD = del /f /q
	RM_DIR = rmdir /s /q
else
	LDFLAGS += -lpthread
	DELETE_CMD = rm -fv
	RM_DIR = rm -rf
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	LDFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	# No additional flags needed for macOS
endif

# Default target
.DEFAULT_GOAL := all

all: $(EXECUTABLE) $(API_EXECUTABLE) $(DEVICE_CHECK_EXEC) $(TEST_EXECUTABLES)

# Build the main executable
$(EXECUTABLE): $(OBJECTS) $(DEPS)
	$(CXX) $(OBJECTS) $(filter-out %.spv, $(DEPS)) -o $@ $(LDFLAGS)

# Build the API executable
$(API_EXECUTABLE): $(OBJECTS) $(DEPS)
	$(CXX) $(filter-out $(BUILD_DIR)/dllama.o,$(OBJECTS)) $(BUILD_DIR)/dllama-api.o $(filter-out %.spv, $(DEPS)) -o $@ $(LDFLAGS)

# Build the device-check executable
$(DEVICE_CHECK_EXEC): $(DEVICE_CHECK_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Build test executables
nn-cpu-test: $(BUILD_DIR)/nn/nn-cpu-test.o $(BUILD_DIR)/nn/nn-quants.o $(BUILD_DIR)/nn/nn-core.o $(BUILD_DIR)/nn/nn-executor.o $(BUILD_DIR)/nn/llamafile/sgemm.o $(BUILD_DIR)/nn/nn-cpu-ops.o $(BUILD_DIR)/nn/nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

nn-cpu-ops-test: $(BUILD_DIR)/nn/nn-cpu-ops-test.o $(BUILD_DIR)/nn/nn-quants.o $(BUILD_DIR)/nn/nn-core.o $(BUILD_DIR)/nn/nn-executor.o $(BUILD_DIR)/nn/llamafile/sgemm.o $(BUILD_DIR)/nn/nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

ifdef DLLAMA_VULKAN
nn-vulkan-test: $(BUILD_DIR)/nn/nn-vulkan-test.o $(BUILD_DIR)/nn/nn-quants.o $(BUILD_DIR)/nn/nn-core.o $(BUILD_DIR)/nn/nn-executor.o $(BUILD_DIR)/nn/nn-vulkan.o $(DEPS)
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LDFLAGS)
endif

tokenizer-test: $(BUILD_DIR)/tokenizer-test.o $(BUILD_DIR)/nn/nn-quants.o $(BUILD_DIR)/nn/nn-core.o $(BUILD_DIR)/nn/llamafile/sgemm.o $(BUILD_DIR)/nn/nn-cpu-ops.o $(BUILD_DIR)/tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile source files to object files with dependency tracking
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) $(BUILD_DIR)/nn $(BUILD_DIR)/nn/llamafile $(BUILD_DIR)/nn/vulkan
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Vulkan shaders
ifdef DLLAMA_VULKAN
$(BUILD_DIR)/nn/vulkan/%.spv: $(VULKAN_SHADER_DIR)/%.comp | $(BUILD_DIR)/nn/vulkan
	$(CGLSLC) $< -o $@
endif

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/nn:
	mkdir -p $(BUILD_DIR)/nn

$(BUILD_DIR)/nn/llamafile:
	mkdir -p $(BUILD_DIR)/nn/llamafile

$(BUILD_DIR)/nn/vulkan:
	mkdir -p $(BUILD_DIR)/nn/vulkan

# Test target
test: $(TEST_EXECUTABLES)
	@echo "Running tests..."
	@for test in $(TEST_EXECUTABLES); do \
		./$$test || exit 1; \
	done

# Install target
install: all
	mkdir -p /usr/local/bin
	cp $(EXECUTABLE) $(API_EXECUTABLE) $(DEVICE_CHECK_EXEC) /usr/local/bin/

# Clean build artifacts
clean:
	$(RM_DIR) $(BUILD_DIR)
	$(DELETE_CMD) $(EXECUTABLE) $(API_EXECUTABLE) $(DEVICE_CHECK_EXEC) $(TEST_EXECUTABLES) *.spv *.o dllama-* socket-benchmark mmap-buffer-* *-test *.exe

.PHONY: all clean test install

# Include dependency files
-include $(DEPS)
