# Copyright 2025 The ZipServ Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include/ 
DEPS_DIR  := ../deps
INCLUDES  += -I$(DEPS_DIR)/safetensors-cpp
INCLUDES  += -I$(DEPS_DIR)/nlohmann-json/single_include
LIBRARIES := -lcublas 
LIBRARIES += -lcusparse 

ifneq ($(MY_PATH),)
INCLUDES  += -I${MY_PATH}/build
LIBRARIES += -L${MY_PATH}/build -lL_API
endif
################################################################################

# Gencode arguments
SMS ?= 80 86 89
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++14
ALL_CCFLAGS += -maxrregcount=255 
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################
# Target rules
all: test_mm

test_mm.o: test_mm.cu utils.h safetensors_weight_loader.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

safetensors_weight_loader.o: safetensors_weight_loader.cpp safetensors_weight_loader.h
	$(EXEC) $(HOST_COMPILER) -O2 -std=c++14 $(INCLUDES) -o $@ -c $<

test_mm: test_mm.o safetensors_weight_loader.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f test_mm test_mm.o safetensors_weight_loader.o llama_weight_loader.o
