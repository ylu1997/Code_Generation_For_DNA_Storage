#ifndef ASSISTANTDATA_CUH
#define ASSISTANTDATA_CUH
#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>
#include <numeric>
#include"bitOperation.cuh"

class h_AssistantArray {
public:
	int batchNum=0;
	int size=0;
	int bitLength=0;
	int numUInt=0;
	unsigned int* masks = nullptr;
	int* shiftAmount = nullptr;
	unsigned int* cache1 = nullptr;
	unsigned int* cache2 = nullptr;
	unsigned int* allOneMask = nullptr;

	h_AssistantArray() {
		masks = NULL;
		shiftAmount = NULL;
		cache1 = NULL;
		cache2 = NULL;
		allOneMask = NULL;
	}

	h_AssistantArray(int batchNum, int size, int bitLength, std::vector<std::vector<bool>>& masksIn, std::vector<int>& shiftAmounts) {
		this->size = size;
		this->batchNum = batchNum;
		this->bitLength = bitLength;
		this->numUInt = (bitLength + 31) / 32;
		masks = new unsigned int[size * numUInt];
		shiftAmount = new int[size];
		cache1 = new unsigned int[batchNum * numUInt];
		cache2 = new unsigned int[batchNum * numUInt];
		allOneMask = new unsigned int[numUInt];
		setOnes(allOneMask, numUInt, bitLength);
		
	}
	~h_AssistantArray() {
		delete[] masks;
		delete[] shiftAmount;
		delete[] cache1;
		delete[] cache2;
		delete[] allOneMask;
	}

	__host__ void printMasks() {
		for (size_t i = 0; i < size; i++)
		{
			unsigned int* mask = masks + i * numUInt;
			printBitArray(mask, bitLength);
			printf("\n");
		}
	}

	__host__ void printShiftAmounts() {
		for (size_t i = 0; i < size; i++)
		{
			printf("%d ", shiftAmount[i]);
		}
		printf("\n");
	}
};

class d_AssistantArray {
public:
	int batchNum;
	int size;
	int bitLength;
	int numUInt;
	unsigned int* masks = nullptr;
	int* shiftAmount = nullptr;
	unsigned int* cache1 = nullptr;
	unsigned int* cache2 = nullptr;
	unsigned int* allOneMask = nullptr;

	d_AssistantArray() {
		masks = NULL;
		shiftAmount = NULL;
		cache1 = NULL;
		cache2 = NULL;
		allOneMask = NULL;
	}

	d_AssistantArray(h_AssistantArray* h_arr) {
		batchNum = h_arr->batchNum;
		size = h_arr->size;
		bitLength = h_arr->bitLength;
		numUInt = h_arr->numUInt;
		cudaMalloc(&masks, size * numUInt * sizeof(unsigned int));
		cudaMemcpy(masks, h_arr->masks, size * numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMalloc(&shiftAmount, size * sizeof(int));
		cudaMemcpy(shiftAmount, h_arr->shiftAmount, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc(&cache1, batchNum * numUInt * sizeof(unsigned int));
		cudaMalloc(&cache2, batchNum * numUInt * sizeof(unsigned int));
		cudaMemcpy(cache1, h_arr->cache1, batchNum * numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(cache2, h_arr->cache2, batchNum * numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMalloc(&allOneMask, numUInt * sizeof(unsigned int));
		cudaMemcpy(allOneMask, h_arr->allOneMask, numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
	}

 
	~d_AssistantArray() {
		cudaFree(masks);
		cudaFree(shiftAmount);
		cudaFree(cache1);
		cudaFree(cache2);
		cudaFree(allOneMask);
	}

	__host__ void copyFromHost(h_AssistantArray& h_array) {
		cudaMalloc(&masks, size * numUInt * sizeof(unsigned int));
		cudaMemcpy(masks, h_array.masks, size * numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMalloc(&shiftAmount, size * sizeof(int));
		cudaMemcpy(shiftAmount, h_array.shiftAmount, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc(&cache1, batchNum * numUInt * sizeof(unsigned int));
		cudaMalloc(&cache2, batchNum * numUInt * sizeof(unsigned int));
		cudaMemcpy(cache1, h_array.cache1, batchNum * numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(cache2, h_array.cache2, batchNum * numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMalloc(&allOneMask, numUInt * sizeof(unsigned int));
		cudaMemcpy(allOneMask, h_array.allOneMask, numUInt * sizeof(unsigned int), cudaMemcpyHostToDevice);
	}

	__host__ void copyToHost(h_AssistantArray& h_array) {
		cudaMemcpy(h_array.masks, masks, size * numUInt * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_array.shiftAmount, shiftAmount, size * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_array.cache1, cache1, batchNum * numUInt * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_array.cache2, cache2, batchNum * numUInt * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_array.allOneMask, allOneMask, numUInt * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	}

	__device__ unsigned int* getMask(int index) {
		//如果index为-1，则返回allOneMask
		if (index == -1) {
			return allOneMask;
		}
		else {
			return masks + index * numUInt;
		}
	}

	__device__ int getShiftAmount(int index) {
		//如果index为-1，则返回0
		if (index == -1) {
			return 0;
		}
		else {
			return shiftAmount[index];
		}
	}

	__device__ unsigned int* getCache1(int index) {
		return cache1 + index * numUInt;
	}

	__device__ unsigned int* getCache2(int index) {
		return cache2 + index * numUInt;
	}

	__device__ unsigned int* getAllOneMask() {
		return allOneMask;
	}

	__host__ __device__ void printMasks() {
		for (size_t i = 0; i < size; i++)
		{
			unsigned int* mask = masks + i * numUInt;
			printBitArray(mask, bitLength);
			printf("\n");
		}
	}

	__host__ __device__ void printShiftAmounts() {
		for (size_t i = 0; i < size; i++)
		{
			printf("%d ", shiftAmount[i]);
		}
		printf("\n");
	}

};

class AssistantArray {
public:
	h_AssistantArray* h_array;
	d_AssistantArray* d_array;
	d_AssistantArray* h2d_array;

	AssistantArray() {
		h_array = NULL;
		d_array = NULL;
		h2d_array = NULL;
	}

	AssistantArray(int batchNum, int size, int bitLength, std::vector<std::vector<bool>>& masksIn, std::vector<int>& shiftAmounts) {
		h_array = new h_AssistantArray(batchNum, size, bitLength, masksIn, shiftAmounts); 
		h2d_array = new d_AssistantArray(h_array);
		cudaMalloc(&d_array, sizeof(d_AssistantArray));
		cudaMemcpy(d_array, h2d_array, sizeof(d_AssistantArray), cudaMemcpyHostToDevice);
	}

	void copyFromHost() {
		h2d_array->copyFromHost(*h_array);
	}

	void copyToHost() {
		h2d_array->copyToHost(*h_array);
	}

	~AssistantArray() {
		delete h_array;
		delete h2d_array;
		cudaFree(d_array);
	}

};


 

#endif // !ASSISTANTDATA_CUH