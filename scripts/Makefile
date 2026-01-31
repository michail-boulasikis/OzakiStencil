CXX = g++
CXXFLAGS = -std=c++20 -fext-numeric-literals -O2 -Wall
LDFLAGS = -lquadmath

all: generate_fp128_tensor

generate_fp128_tensor: generate_fp128_tensor.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f generate_fp128_tensor

.PHONY: all clean
