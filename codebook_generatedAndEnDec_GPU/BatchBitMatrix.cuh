#ifndef BATCHBITMATRIX_CUH
#define BATCHBITMATRIX_CUH
#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>
#include <numeric>
#include"bitOperation.cuh"


__host__ __device__ int getMatrixSize(int* properties) {
	return properties[1] * properties[2];
}

__host__ __device__ int getNumUIntEachCell(int* properties) {
	return properties[4];
}

__host__ __device__ int getNumUIntEachMatrix(int* properties) {
	return getMatrixSize(properties) * getNumUIntEachCell(properties);
}

__host__ __device__ long long getNumUIntAllBatch(int* properties) {
	return ((long long)properties[0]) * getNumUIntEachMatrix(properties);
}

__host__ __device__ unsigned int* getMatrixPointer(int* properties, unsigned int* data, int batchIndex) {
	return data + batchIndex * getNumUIntEachMatrix(properties);
}

__host__ __device__ unsigned int* getCellPointer(int* properties, unsigned int* data, int batchIndex, int matrixRow, int matrixCol) {
	return getMatrixPointer(properties, data, batchIndex) + matrixRow * properties[2] * properties[4] + matrixCol * properties[4];
}

__host__ __device__ void setCell(int* properties, unsigned int* data, int batchIndex, int matrixRow, int matrixCol, unsigned int* otherCell) {
	unsigned int* cell = getCellPointer(properties, data, batchIndex, matrixRow, matrixCol);
	for (size_t i = 0; i < properties[4]; i++)
	{
		cell[i] = otherCell[i];
	}
}

__host__ __device__ void printCell(int* properties, unsigned int* data, int batchIndex, int matrixRow, int matrixCol) {
	unsigned int* cell = getCellPointer(properties, data, batchIndex, matrixRow, matrixCol);
	// 打印bit，不超过bitlength，每8位用空格隔开
	for (size_t i = 0; i < properties[3]; i++)
	{
		printf("%d", (cell[0] >> i) & 1);
		if (i % 8 == 7)
		{
			printf(" ");
		}
	}
}

__host__ __device__ void printMatrix(int* properties, unsigned int* data, int batchIndex) {
	for (size_t i = 0; i < properties[1]; i++)
	{
		printf("|");
		for (size_t j = 0; j < properties[2]; j++)
		{
			printCell(properties, data, batchIndex, i, j);
			printf("|");
		}
		printf("\n");
	}
}

__host__ __device__ void printBatchMatrix(int* properties, unsigned int* data) {
	for (size_t i = 0; i < properties[0]; i++)
	{
		printf("Batch %d\n", i);
		printMatrix(properties, data, i);
	}
}


class h_BatchBitMatrix {
public:
	int* properties;
	unsigned int* data;

	h_BatchBitMatrix(int batchSize, int matrixRow, int matrixCol, int bitLength) {
		properties = new int[5];
		properties[0] = batchSize;
		properties[1] = matrixRow;
		properties[2] = matrixCol;
		properties[3] = bitLength;
		properties[4] = (properties[3] + 31) / 32;
		long long h_size = getNumUIntAllBatch(properties);
		data = new unsigned int[h_size];
		for (size_t i = 0; i < h_size; i++)
		{
			data[i] = 0;
		}
	}

	__host__ void h_setCell(int batchIndex, int matrixRow, int matrixCol, unsigned int* otherCell) {
		setCell(properties, data, batchIndex, matrixRow, matrixCol, otherCell);
	}

	__host__ void h_printCell(int batchIndex, int matrixRow, int matrixCol) {
		printCell(properties, data, batchIndex, matrixRow, matrixCol);
	}

	__host__ void h_printMatrix(int batchIndex) {
		printMatrix(properties, data, batchIndex);
	}

	__host__ void h_printBatchMatrix() {
		printBatchMatrix(properties, data);
	}

	~h_BatchBitMatrix() {
		if (properties != NULL) {
			delete[] properties;
		}
		if (data != NULL) {
			delete[] data;
		}

	}

	__host__ void h_transition(int batchIndex, int row1, int col1, int row2, int col2, unsigned int* mask, int shiftAmount, unsigned int* cache1, unsigned int* cache2) {
		unsigned int* cell1 = getCellPointer(properties, data, batchIndex, row1, col1);
		unsigned int* cell2 = getCellPointer(properties, data, batchIndex, row2, col2);

		setEmpty(cache1, properties[4]);
		setEmpty(cache2, properties[4]);
		bitwiseAnd(cell1, mask, cache1, properties[4]);
		shiftHarray(cache1, cache2, properties[4], shiftAmount, properties[3]);
		bitwiseOr(cache2, cell2, cache1, properties[4]);
		cloneArray(cache1, cell2, properties[4]);
	}

	__host__ int h_getNumBatch() {
		return properties[0];
	}

	__host__ int h_getNumRows() {
		return properties[1];
	}

	__host__ int h_getNumCols() {
		return properties[2];
	}

	__host__ int h_getBitLength() {
		return properties[3];
	}

	__host__ int h_getNumUInt() {
		return properties[4];
	}

	__host__ unsigned int* h_getCellPointer(int batchIndex, int matrixRow, int matrixCol) {
		return getCellPointer(properties, data, batchIndex, matrixRow, matrixCol);
	}

};

class d_BatchBitMatrix {
public:
	int* properties;
	unsigned int* data;

	__host__ d_BatchBitMatrix() {
		properties = NULL;
		data = NULL;
	}

	__host__ d_BatchBitMatrix(h_BatchBitMatrix* mat) {
		cudaMalloc(&properties, 5 * sizeof(int));
		cudaMemcpy(properties, mat->properties, 5 * sizeof(int), cudaMemcpyHostToDevice);
		long long h_size = getNumUIntAllBatch(mat->properties);
		cudaMalloc(&data, h_size * sizeof(unsigned int));
		cudaMemcpy(data, mat->data, h_size * sizeof(unsigned int), cudaMemcpyHostToDevice);

	}

	__host__ ~d_BatchBitMatrix() {
		cudaFree(properties);
		cudaFree(data);
	}

	__host__ void copyFromHost(h_BatchBitMatrix* mat) {
		if (properties != NULL) {
			cudaFree(properties);
		}
		if (data != NULL)
		{
			cudaFree(data);
		}
		cudaMemcpy(properties, mat->properties, 5 * sizeof(int), cudaMemcpyHostToDevice);
		long long h_size = getNumUIntAllBatch(mat->properties);
		cudaMemcpy(data, mat->data, h_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	}


	__host__ void copyToHost(h_BatchBitMatrix* mat) {
		cudaMemcpy(mat->properties, properties, 5 * sizeof(int), cudaMemcpyDeviceToHost);
		long long h_size = getNumUIntAllBatch(mat->properties);
		cudaMemcpy(mat->data, data, h_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	}

	__device__ void d_setCell(int batchIndex, int matrixRow, int matrixCol, unsigned int* otherCell) {
		// 调用上面的函数 
		setCell(properties, data, batchIndex, matrixRow, matrixCol, otherCell);
	}

	__device__ void d_printCell(int batchIndex, int matrixRow, int matrixCol) {
		printCell(properties, data, batchIndex, matrixRow, matrixCol);
	}

	__device__ void d_printMatrix(int batchIndex) {
		printMatrix(properties, data, batchIndex);
	}

	__device__ void d_printBatchMatrix() {
		printBatchMatrix(properties, data);
	}

	__device__ void d_transition(int batchIndex, int row1, int col1, int row2, int col2, unsigned int* mask, int shiftAmount, unsigned int* cache1, unsigned int* cache2) {
		unsigned int* cell1 = getCellPointer(properties, data, batchIndex, row1, col1);
		unsigned int* cell2 = getCellPointer(properties, data, batchIndex, row2, col2);
		setEmpty(cache1, properties[4]);
		setEmpty(cache2, properties[4]);
		bitwiseAnd(cell1, mask, cache1, properties[4]);
		shiftHarray(cache1, cache2, properties[4], shiftAmount, properties[3]);
		bitwiseOr(cache2, cell2, cache1, properties[4]);
		cloneArray(cache1, cell2, properties[4]);
	}

	__device__ int d_getNumBatch() {
		return properties[0];
	}

	__device__ int d_getNumRows() {
		return properties[1];
	}

	__device__ int d_getNumCols() {
		return properties[2];
	}

	__device__ int d_getBitLength() {
		return properties[3];
	}

	__device__ int d_getNumUInt() {
		return properties[4];
	}

	__device__ unsigned int* d_getCellPointer(int batchIndex, int matrixRow, int matrixCol) {
		return getCellPointer(properties, data, batchIndex, matrixRow, matrixCol);
	}

};

class BatchBitMatrix {
public:
	h_BatchBitMatrix* h_mat;
	d_BatchBitMatrix* h2d_mat;
	d_BatchBitMatrix* d_mat;

	BatchBitMatrix() {
		h_mat = NULL;
		h2d_mat = NULL;
		d_mat = NULL;
	}

	BatchBitMatrix(int batchSize, int matrixRow, int matrixCol, int bitLength) {
		h_mat = new h_BatchBitMatrix(batchSize, matrixRow, matrixCol, bitLength);
		h2d_mat = new d_BatchBitMatrix(h_mat);
		cudaMalloc(&d_mat, sizeof(d_BatchBitMatrix));
		cudaMemcpy(d_mat, h2d_mat, sizeof(d_BatchBitMatrix), cudaMemcpyHostToDevice);
	}

	void copyFromHost() {
		h2d_mat->copyFromHost(h_mat);
	}

	void copyToHost() {
		h2d_mat->copyToHost(h_mat);
	}

	~BatchBitMatrix() {
		if (h_mat != NULL) {
			delete h_mat;
		}
		if (h2d_mat != NULL) {
			delete h2d_mat;
		}
		if (d_mat != NULL) {
			cudaFree(d_mat);
		}

	}
};

#endif // !BATCHBITMATRIX_CUH
