TEST_SOURCE = cudatest.cu

TARGETBIN := ./cudatest

CC = nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN)  -rdc=true -std=c++11

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
