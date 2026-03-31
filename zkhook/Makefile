# Compilers
# Auto-detect nvcc path: prefer nvcc in system PATH
NVCC := $(shell which nvcc 2>/dev/null || echo "nvcc")

# Include and library paths
# If CUDA_HOME is set, use it; otherwise try CONDA_PREFIX; finally use system default
ifdef CUDA_HOME
    INCLUDES := -I$(CUDA_HOME)/include
    LIBS := -L$(CUDA_HOME)/lib64
else ifdef CONDA_PREFIX
    INCLUDES := -I$(CONDA_PREFIX)/include
    LIBS := -L$(CONDA_PREFIX)/lib
else
    INCLUDES := -I/usr/local/cuda/include
    LIBS := -L/usr/local/cuda/lib64
endif

# GPU architecture - modify according to your GPU:
# RTX 30xx (Ampere) -> sm_86
# RTX 40xx (Ada)    -> sm_89
# A100              -> sm_80
# V100              -> sm_70
ARCH := sm_89

# NVCC compiler flags
NVCC_FLAGS := -arch=$(ARCH) -std=c++17

# Source and object files
CU_SRCS := bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu sha256.cu proof_stream.cu attn_proof.cu zkrelu.cu ffn_proof.cu
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_SRCS := $(wildcard *.cpp)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)

# Main targets: Complete Transformer layer ZK proof (integrated)
# table_gen: Generate Softmax and SwiGLU lookup tables (shared by P and V)
# transformer_prove: Generate complete proof (integrated)
# transformer_verify: Verify proof (integrated)
TARGETS := table_gen transformer_prove transformer_verify

# Stepwise execution targets (consistent with baseline approach)
# rmsnorm_prove/verify: RMSNorm layer
# attn_prove/verify: Self-Attention layer
# ffn_prove/verify: FFN layer
STEPWISE_TARGETS := rmsnorm_prove rmsnorm_verify attn_prove attn_verify ffn_prove ffn_verify

ALL_TARGETS := $(TARGETS) $(STEPWISE_TARGETS)

TARGET_OBJS := $(ALL_TARGETS:=.o)

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# Pattern rule for C++ source files
%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc -dlto $< -o $@

# General rule for building each target
$(ALL_TARGETS): % : %.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@ -dlto

# Clean rule
clean:
	rm -f $(TARGET_OBJS) $(CU_OBJS) $(CPP_OBJS) $(ALL_TARGETS)

# Default rule - only compile main targets
all: $(TARGETS)

# Compile stepwise execution targets
stepwise: $(STEPWISE_TARGETS) table_gen

# Compile all targets
all-targets: $(ALL_TARGETS)

# Help information
help:
	@echo "zkLLM Complete Transformer Layer Zero-Knowledge Proof"
	@echo "======================================================"
	@echo ""
	@echo "Main targets (integrated execution):"
	@echo "  table_gen          - Generate Softmax/SwiGLU lookup tables"
	@echo "  transformer_prove  - Generate complete Transformer layer ZK proof"
	@echo "  transformer_verify - Verify complete Transformer layer ZK proof"
	@echo ""
	@echo "Stepwise execution targets (consistent with baseline):"
	@echo "  rmsnorm_prove/verify - RMSNorm layer prove/verify"
	@echo "  attn_prove/verify    - Self-Attention layer prove/verify"
	@echo "  ffn_prove/verify     - FFN layer prove/verify"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make all           - Compile integrated targets"
	@echo "  make stepwise      - Compile stepwise execution targets"
	@echo "  make all-targets   - Compile all targets"
	@echo "  make clean         - Clean compiled files"
	@echo ""
	@echo "Benchmarks:"
	@echo "  python ../benchmark_llama7b_actual.py    - Integrated test"
	@echo "  python ../benchmark_stepwise.py          - Stepwise execution test"
	@echo ""
	@echo "Zero-Knowledge Guarantees:"
	@echo "  - Input X: Commitment (hidden from verifier)"
	@echo "  - Weights W: Commitment (hidden from verifier)"
	@echo "  - Output Y: Commitment (hidden from verifier)"
