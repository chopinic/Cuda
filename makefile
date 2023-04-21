SOURCE_FILE_LOOP = number_crunching_loop.cu
TARGETBIN_LOOP := ./loopgpu

SOURCE_FILE_TASK = number_crunching_task.cu
TARGETBIN_TASK := ./taskgpu

CC = nvcc

loopgpu: $(SOURCE_FILE_LOOP)
	$(CC)  $(SOURCE_FILE_LOOP) -O0 -o $(TARGETBIN_LOOP)  -rdc=true -std=c++11


taskgpu: $(SOURCE_FILE_TASK)
	$(CC)  $(SOURCE_FILE_TASK) -O0 -o $(TARGETBIN_TASK)  -rdc=true -std=c++11


.PHONY:clean
clean:
	-rm -rf $(TARGETBIN_LOOP)
	-rm -rf $(TARGETBIN_TASK)
