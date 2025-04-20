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
# Include sources from src/ and src/nn/
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/nn/*.cpp)
# Map src/*.cpp to build/*.o and src/nn/*.cpp to build/nn/*.o
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS = $(OBJECTS:.o=.d)
EXECUTABLE = dllama

# Device check
DEVICE_CHECK_SRC = device-check.cpp
DEVICE_CHECK_EXEC = device-check

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

all: $(EXECUTABLE) $(DEVICE_CHECK_EXEC)

# Build the main executable
$(EXECUTABLE): $(OBJECTS) $(DEPS)
	$(CXX) $(OBJECTS) $(filter-out %.spv, $(DEPS)) -o $@ $(LDFLAGS)

# Build the device-check executable
$(DEVICE_CHECK_EXEC): $(DEVICE_CHECK_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Compile source files to object files with dependency tracking
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Compile source files in src/nn/ to object files in build/nn/
$(BUILD_DIR)/nn/%.o: $(SRC_DIR)/nn/%.cpp | $(BUILD_DIR)/nn
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

$(BUILD_DIR)/nn/vulkan:
	mkdir -p $(BUILD_DIR)/nn/vulkan

# Test target (placeholder for test files)
test:
	@echo "Running tests..."
	@for test in $(wildcard *-test); do \
		./$$test || exit 1; \
	done

# Install target
install: all
	mkdir -p /usr/local/bin
	cp $(EXECUTABLE) $(DEVICE_CHECK_EXEC) /usr/local/bin/

# Clean build artifacts
clean:
	$(RM_DIR) $(BUILD_DIR)
	$(DELETE_CMD) $(EXECUTABLE) $(DEVICE_CHECK_EXEC) *.spv *.o dllama-* socket-benchmark mmap-buffer-* *-test *.exe

.PHONY: all clean test install

# Include dependency files
-include $(DEPS)
