#ifndef BITOPERATION_CUH
#define BITOPERATION_CUH
#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>
#include <numeric>

// bitwise operate part//
__host__ __device__ int getIndex(int pos, int bitSize)
{
	return pos / bitSize;
}

__host__ __device__ int getOffset(int pos, int bitSize)
{
	return pos % bitSize;
}

__host__ __device__ void cleanOverBits(unsigned int* arr, int bitLength) {
	//将超过bitlength的位清零
	int index = getIndex(bitLength, 32);
	int offset = getOffset(bitLength, 32);
	if (offset != 0)
	{
		arr[index] &= (1 << offset) - 1;
	}
}

__host__ __device__ void setBit(unsigned int* arr, int bitLength, int index, bool val) {
	int i = getIndex(index, 32);
	int j = getOffset(index, 32);
	if (val)
	{
		arr[i] |= 1 << j;
	}
	else
	{
		arr[i] &= ~(1 << j);
	}
}

__host__ __device__ bool getBit(unsigned int* arr, int bitLength, int index) {
	int i = getIndex(index, 32);
	int j = getOffset(index, 32);
	return (arr[i] >> j) & 1;
}

__host__ __device__ void bitwiseAnd(unsigned int* arr1, unsigned int* arr2, unsigned int* result, int numUInt) {
	for (size_t i = 0; i < numUInt; i++)
	{
		result[i] = arr1[i] & arr2[i];
	}
}

__host__ __device__ void bitwiseOr(unsigned int* arr1, unsigned int* arr2, unsigned int* result, int numUInt) {
	for (size_t i = 0; i < numUInt; i++)
	{
		result[i] = arr1[i] | arr2[i];
	}
}

__host__ __device__ void bitwiseInvert(unsigned int* arr, unsigned int* result, int numUInt, int bitLength) {
	for (size_t i = 0; i < numUInt; i++)
	{
		result[i] = ~arr[i];
	}
	cleanOverBits(result, bitLength);
}

__host__ __device__ bool bitwiseEqual(unsigned int* arr1, unsigned int* arr2, int numUInt)
{
	for (size_t i = 0; i < numUInt; i++)
	{
		if (arr1[i] != arr2[i])
		{
			return false;
		}
	}
	return true;
}

__host__ __device__ void cloneArray(unsigned int* arr, unsigned int* result, int numUInt) {
	for (size_t i = 0; i < numUInt; i++)
	{
		result[i] = arr[i];
	}
}

__host__ __device__ void shiftHarray(unsigned int* arr, unsigned int* result, int len, int shiftAmount, int bitLength) {
	const int index = getIndex(shiftAmount, 32);
	const int offset = getOffset(shiftAmount, 32);
	const int n = len - index - 1;
	const int bound_offset = (8 * sizeof(int) * len - bitLength);
	// Perform shifting
	if (offset != 0) {
		for (int i = 0; i < n; i++) {
			result[len - 1 - i] = (arr[len - 1 - (index + i)] << offset);
			result[len - 1 - i] |= (arr[len - 1 - (index + i + 1)] >> (sizeof(int) * 8 - offset));
		}
		result[len - 1 - n] = (arr[len - 1 - (index + n)] << offset);
	}
	else {
		for (int i = 0; i < n; i++) {
			result[len - 1 - i] = arr[len - 1 - (index + i)];
		}
		result[len - 1 - n] = arr[len - 1 - (index + n)];
	}

	cleanOverBits(result, bitLength);
}

__host__ __device__ void shiftLarray(unsigned int* arr, unsigned int* result, int len, int shiftAmount, int bitLength) {
	const int index = getIndex(shiftAmount, 32);
	const int offset = getOffset(shiftAmount, 32);
	const int n = len - index - 1;
	const int bound_offset = (8 * sizeof(int) * len - bitLength);
	// Perform shifting
	if (offset != 0) {
		for (int i = 0; i < n; i++) {
			result[i] = (arr[index + i] >> offset);
			result[i] |= (arr[index + i + 1] << (sizeof(int) * 8 - offset));
		}
		result[n] = (arr[index + n] >> offset);
	}
	else {
		for (int i = 0; i < n; i++) {
			result[i] = arr[index + i];
		}
		result[n] = arr[index + n];
	}
}

__host__ __device__ void setEmpty(unsigned int* arr, int numUInt) {
	for (size_t i = 0; i < numUInt; i++)
	{
		arr[i] = 0;
	}
}

__host__ __device__ void setOnes(unsigned int* arr, int numUInt, int bitLength) {
	//在bitlength范围内，将所有的位都设置为1。先取反，在高位清零
	bitwiseInvert(arr, arr, numUInt, bitLength);
}

__host__ __device__ void setOne(unsigned int* arr, int bitLength) {
	setBit(arr, bitLength, 0, 1);
}

__host__ __device__ void printBitArray(unsigned int* arr, int bitLength) {
	for (size_t i = 0; i < bitLength; i++)
	{
		printf("%d", (arr[0] >> i) & 1);
		if (i % 8 == 7)
		{
			printf(" ");
		}
	}
}

__host__ __device__ void printBitData(unsigned int* arr, int bitLength)
{
	for (size_t i = 0; i < bitLength; i++)
	{
		printf("%d", (arr[0] >> i) & 1);
		if (i % 8 == 7)
		{
			printf(" ");
		}
	}
}

//#######################//

#endif // !BITOPERATION_CUH
