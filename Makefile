

# GPU 架构（默认 sm_60，对应 Tesla P100）
ARCH ?= sm_60


HOST_COMP ?= mpicc



NVCC    ?= nvcc               
TARGET  := shishi1            
SRC     := scuda1.cu           

# 通用编译选项
COMMONFLAGS := -O3 -std=c++11

# nvcc 编译 / 链接选项
NVCCFLAGS  := $(COMMONFLAGS) -arch=$(ARCH) -ccbin=$(HOST_COMP)
LDFLAGS    := -lmpi -lm  -lstdc++         # 显式链接 MPI 库（你之前就是这么干的）



all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
