#ifndef CUDABITCLASS_CUH
#define CUDABITCLASS_CUH

#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>

class cudaBitArray {
private:
	unsigned int* data;
	int numUInt;
	int bitLength;
public:
	cudaBitArray() {
		data = nullptr;
		numUInt = 0;
		bitLength = 0;
	}

	cudaBitArray(int n) {
		numUInt = (n + 31) / 32;
		bitLength = n;
		cudaMallocManaged(&data, numUInt * sizeof(unsigned int));
		for (int i = 0; i < numUInt; i++) {
			data[i] = 0;
		}
	}

	cudaBitArray(const std::vector<bool>& input) {
		numUInt = (input.size() + 31) / 32;
		bitLength = input.size();
		cudaMallocManaged(&data, numUInt * sizeof(unsigned int));
		setEmpty();
		for (int i = 0; i < input.size(); i++) {
			setBit(i, input[i]);
		}
	}


	~cudaBitArray() {
		cudaFree(data);
	}

	__host__ __device__ void setBit(int i, bool value) {
		if (value) {
			data[i / 32] |= (1 << (i % 32));
		}
		else {
			data[i / 32] &= ~(1 << (i % 32));
		}
	}

	__host__ __device__ bool getBit(int i) const {
		return data[i / 32] & (1 << (i % 32));
	}

	__host__ __device__ int len() const {
		return bitLength;
	}

	__host__ __device__ void setItem(int index, unsigned int item) {
		data[index] = item;
	}

	__host__ __device__ unsigned int getItem(int index) {
		return (index >= 0 && index < numUInt) ? data[index] : 0;
	}

	__host__ __device__ void setArray(cudaBitArray* arr) {
		for (int i = 0; i < numUInt; i++) {
			data[i] = arr->data[i];
		}
	}

	__host__ __device__ void setEmpty() {
		for (int i = 0; i < numUInt; i++) {
			data[i] = 0;
		}
	}

	__host__ __device__ void bitwiseAnd(cudaBitArray* other, cudaBitArray* result) {
		for (int i = 0; i < numUInt; i++) {
			result->data[i] = data[i] & other->data[i];
		}
	}

	__host__ __device__ void bitwiseOr(cudaBitArray* other, cudaBitArray* result) {
		for (int i = 0; i < numUInt; i++) {
			result->data[i] = data[i] | other->data[i];
		}
	}

	__host__ __device__ bool isEqual(cudaBitArray* other) {
		for (int i = 0; i < numUInt; i++) {
			if (data[i] != other->data[i]) {
				return false;
			}
		}
		return true;
	}

	__host__ __device__ void bitwiseInvert(cudaBitArray* result) {
		for (int i = 0; i < numUInt; i++) {
			result->data[i] = ~data[i];
		}
		//超过部分设置为0，通过向高位平移事先
		result->data[numUInt - 1] &= (0xFFFFFFFF >> (32 - bitLength % 32));
	}

	__host__ __device__ void bitwiseShiftL(cudaBitArray* result, int shiftAmount) {
		const int index = shiftAmount / 32;
		const int offset = shiftAmount % 32;
		int n = numUInt - index - 1;
		result->setEmpty();
		if (offset != 0) {
			for (int i = 0; i < n; i++) {
				result->data[i] = (data[index + i] >> offset);
				result->data[i] |= (data[index + i + 1] << (32 - offset));
			}
			result->data[n] = (data[index + n] >> offset);
		}
		else {
			for (int i = 0; i < n; i++) {
				result->data[i] = data[index + i];
			}
			result->data[n] = data[index + n];
		}
	}

	__host__ __device__ void bitwiseShiftH(cudaBitArray* result, int shiftAmount) {
		const int index = shiftAmount / 32;
		const int offset = shiftAmount % 32;
		const int n = numUInt - index - 1;
		const int bound_offset = (32 * numUInt - bitLength);
		result->setEmpty();
		if (offset != 0) {
			for (int i = 0; i < n; i++) {
				result->data[numUInt - 1 - i] = (data[numUInt - 1 - (index + i)] << offset);
				result->data[numUInt - 1 - i] |= (data[numUInt - 1 - (index + i + 1)] >> (32 - offset));
			}
			result->data[numUInt - 1 - n] = (data[numUInt - 1 - (index + n)] << offset);
		}
		else {
			for (int i = 0; i < n; i++) {
				result->data[numUInt - 1 - i] = data[numUInt - 1 - (index + i)];
			}
			result->data[numUInt - 1 - n] = data[numUInt - 1 - (index + n)];
		}

		if (bound_offset != 0) {
			result->data[numUInt - 1] = (result->data[numUInt - 1] << bound_offset) >> bound_offset;
		}
	}

	__host__ __device__ void print() const {
		for (int i = 0; i < bitLength; i++) {
			/*if (data[i / 32] & (1 << (i % 32))) {
				std::cout << "1";
			}
			else {
				std::cout << "0";
			}
			if ((i + 1) % 8 == 0) {
				std::cout << " ";
			}
		}
		std::cout << std::endl;*/
			if (data[i / 32] & (1 << (i % 32))) {
				printf("1");
			}
			else {
				printf("0");
			}
			if ((i + 1) % 8 == 0) {
				printf(" ");
			}
		} 
	}

	std::string toString() {
		std::string str;
		for (int i = 0; i < bitLength; i++) {
			if (data[i / 32] & (1 << (i % 32))) {
				str += "1";
			}
			else {
				str += "0";
			}
			if ((i + 1) % 8 == 0) {
				str += " ";
			}
		}
		return str;
	}

	__host__ void clone(cudaBitArray* result) const {
		cudaMemcpy(result->data, data, numUInt * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		result->bitLength = bitLength;
		result->numUInt = numUInt;
	}

};

class cudaBitVector {
private:
	cudaBitArray* data;
	int numArray;
	int bitLength;
public:
	cudaBitVector() {
		data = nullptr;
		numArray = 0;
		bitLength = 0;
	}

	cudaBitVector(int n, int m) {
		numArray = n;
		bitLength = m;
		cudaMallocManaged(&data, numArray * sizeof(cudaBitArray));
		for (int i = 0; i < numArray; i++) {
			new(data + i) cudaBitArray(m);
		}
	}

	cudaBitVector(std::vector<std::vector<bool>>& input) {
		numArray = input.size();
		bitLength = input[0].size();
		cudaMallocManaged(&data, numArray * sizeof(cudaBitArray));
		for (int i = 0; i < numArray; i++) {
			new(data + i) cudaBitArray(input[i]);
		}
	}


	~cudaBitVector() {
		for (int i = 0; i < numArray; i++) {
			data[i].~cudaBitArray();
		}

		cudaFree(data);
	}

	__host__ __device__ void setBit(int i, int j, bool value) {
		data[i].setBit(j, value);
	}

	__host__ __device__ bool getBit(int i, int j) const {
		return data[i].getBit(j);
	}

	__host__ __device__ int lenVec() const {
		return numArray;
	}

	__host__ __device__ int lenBit() const {
		return bitLength;
	}

	__host__ __device__ void setEmpty() {
		for (int i = 0; i < numArray; i++) {
			data[i].setEmpty();
		}
	}

	__host__ __device__ cudaBitArray* getArray(int i) {
		return (i >= 0 && i < numArray) ? data + i : nullptr;
	}

	__host__ __device__ void setArray(int i, cudaBitArray* arr) {
		data[i].setArray(arr);
	}

	__host__ __device__ void setVector(cudaBitVector* vec) {
		for (int i = 0; i < numArray; i++) {
			data[i].setArray(vec->getArray(i));
		}
	}

	void print() const {
		for (int i = 0; i < numArray; i++) {
			std::cout << "Array " << i << ": ";
			data[i].print();
			std::cout << std::endl;
		}
	}

	__host__ void clone(cudaBitVector* result) const {
		for (int i = 0; i < numArray; i++) {
			data[i].clone(result->getArray(i));
		}
		result->numArray = numArray;
		result->bitLength = bitLength;
	}
};

class cudaBitMatrix {
private:
	cudaBitArray* data;
	int numRows;
	int numCols;
	int bitLength;
public:
	cudaBitMatrix() {
		data = nullptr;
		numRows = 0;
		numCols = 0;
		bitLength = 0;
	}

	cudaBitMatrix(int n, int m, int k) {
		numRows = n;
		numCols = m;
		bitLength = k;
		cudaMallocManaged(&data, numRows * numCols * sizeof(cudaBitArray));
		for (int i = 0; i < numRows * numCols; i++) {
			new(data + i) cudaBitArray(k);
		}
	}

	~cudaBitMatrix() {
		for (int i = 0; i < numRows * numCols; i++) {
			data[i].~cudaBitArray();
		}
		cudaFree(data);
	}

	__host__ __device__ cudaBitArray* getCell(int row, int col) const {
		return(row >= 0 && row < numRows && col >= 0 && col < numCols) ? data + row * numCols + col : nullptr;
	}

	__host__ __device__ int getNumRows() const {
		return numRows;
	}

	__host__ __device__ int getNumCols() const {
		return numCols;
	}

	__host__ __device__ int getBitLength() const {
		return bitLength;
	}

	__host__ __device__ void setEmpty() {
		for (int i = 0; i < numRows * numCols; i++) {
			data[i].setEmpty();
		}
	}

	__host__ __device__ void setCell(int row, int col, cudaBitArray* arr) {
		data[row * numCols + col].setArray(arr);
	}

	__host__ __device__ void transition(int row1, int col1, int row2, int col2,
		cudaBitArray* mask, int shiftAmount, cudaBitArray* cache1, cudaBitArray* cache2) {
		cache1->setEmpty();
		cache2->setEmpty();
		getCell(row1, col1)->bitwiseAnd(mask, cache1);
		cache1->bitwiseShiftH(cache2, shiftAmount);
		getCell(row2, col2)->bitwiseOr(cache2, cache1);
		setCell(row2, col2, cache1);
	}

	__host__ __device__ void setBit(int row, int col, int i, bool val) {
		getCell(row, col)->setBit(i, val);
	}

	__host__ __device__ void initMiddleCell() {
		setBit(0, getNumCols() / 2, 0, true);
	}

	__host__ __device__ void print() const {
		for (int i = 0; i < numRows; i++) {
			//std::cout << "| ";
			//for (int j = 0; j < numCols; j++) {
			//	std::cout << getCell(i, j)->toString() << "| ";
			//}
			//std::cout << std::endl;
			printf("| ");
			for (int j = 0; j < numCols; j++) {
				getCell(i, j)->print();
				printf("| ");
			}
			printf("\n");
		}
	}

	__host__ void clone(cudaBitMatrix* result) const {
		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numCols; j++) {
				getCell(i, j)->clone(result->getCell(i, j));
			}
		}
		result->numRows = numRows;
		result->numCols = numCols;
		result->bitLength = bitLength;
	}
};

class cudaBatchBitMatrix {
private:
	cudaBitMatrix* data;
	int numMatrices;
	int numRows;
	int numCols;
	int bitLength;
public:
	cudaBatchBitMatrix() {
		data = nullptr;
		numMatrices = 0;
		numRows = 0;
		numCols = 0;
		bitLength = 0;
	}

	cudaBatchBitMatrix(int n, int m, int k, int l) {
		numMatrices = n;
		numRows = m;
		numCols = k;
		bitLength = l;
		cudaMallocManaged(&data, numMatrices * sizeof(cudaBitMatrix));
		for (int i = 0; i < numMatrices; i++) {
			new(data + i) cudaBitMatrix(m, k, l);
		}
	}

	~cudaBatchBitMatrix() {
		for (int i = 0; i < numMatrices; i++) {
			data[i].~cudaBitMatrix();
		}
		cudaFree(data);
	}

	void print() {
		for (int i = 0; i < numMatrices; i++) {
			std::cout << "Matrix " << i << std::endl;
			data[i].print();
		}
	}

	__host__ __device__ cudaBitMatrix* getMatrix(int i) {
		return (i >= 0 && i < numMatrices) ? data + i : nullptr;
	}

	__host__ void clone(cudaBatchBitMatrix* result) const {
		for (int i = 0; i < numMatrices; i++) {
			data[i].clone(result->getMatrix(i));
		}
		result->numMatrices = numMatrices;
		result->numRows = numRows;
		result->numCols = numCols;
		result->bitLength = bitLength;
	}

	__host__ __device__ int getNumMatrices() const {
		return numMatrices;
	}

	__host__ __device__ int getNumRows() const {
		return numRows;
	}

	__host__ __device__ int getNumCols() const {
		return numCols;
	}

	__host__ __device__ int getBitLength() const {
		return bitLength;
	}

	__host__ void initMiddleCell() {
		for (int i = 0; i < numMatrices; i++) {
			data[i].initMiddleCell();
		}
	}

};

static void testBitArrayClass1() {
	std::vector<bool> input = { 1, 0, 1, 0, 1, 0, 1, 0, 0, 0 };
	cudaBitArray* bitArray = new cudaBitArray(input);
	cudaBitArray* result = new cudaBitArray(10);
	bitArray->print();
	bitArray->bitwiseInvert(result);
	result->print();
	for (int i = 0; i < 5; i++) {
		bitArray->bitwiseShiftL(result, i);
		result->print();
		bitArray->bitwiseShiftH(result, i);
		result->print();
	}

	delete result;
	delete bitArray;
}

static void testBitArrayClass2() {
	std::vector<bool> input = { 1, 0, 1, 0, 1, 0, 1, 0, 0, 0 };
	cudaBitArray* bitArray = new cudaBitArray(input);
	cudaBitArray* result = new cudaBitArray(10);

	std::cout << "Original BitArray: ";
	bitArray->print();

	std::cout << "Testing bitwiseShiftL:" << std::endl;
	for (int i = 0; i < 10; i++) {
		bitArray->bitwiseShiftL(result, i);
		std::cout << "Shift Amount " << i << ": ";
		result->print();
	}

	std::cout << "Testing bitwiseShiftH:" << std::endl;
	for (int i = 0; i < 10; i++) {
		bitArray->bitwiseShiftH(result, i);
		std::cout << "Shift Amount " << i << ": ";
		result->print();
	}

	delete result;
	delete bitArray;
}

__global__ void kernelTestBitVector(cudaBitVector* bitVector, cudaBitVector* result) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < bitVector->lenVec()) {
		cudaBitArray* bitArray = bitVector->getArray(idx);
		cudaBitArray* resultArray = result->getArray(idx);
		bitArray->setEmpty();
		bitArray->setBit(0, true);
		bitArray->bitwiseShiftH(resultArray, idx + 30);
	}
}

static void testBitVector() {
	int numArrays = 20;
	int bitLength = 66;
	cudaBitVector* bitVector;
	cudaMallocManaged(&bitVector, sizeof(cudaBitVector));
	new(bitVector) cudaBitVector(numArrays, bitLength);
	cudaBitVector* result;
	cudaMallocManaged(&result, sizeof(cudaBitVector));
	new(result) cudaBitVector(numArrays, bitLength);

	std::cout << "Original BitVector:" << std::endl;
	bitVector->print();

	// 调用 kernel 函数
	kernelTestBitVector << <1, numArrays >> > (bitVector, result);
	cudaDeviceSynchronize();

	std::cout << "After kernelTestBitVector:" << std::endl;
	bitVector->print();
	std::cout << "Result BitVector:" << std::endl;
	result->print();
	cudaFree(bitVector);
}

static void testBitMatrix() {
	int numRows = 4;
	int numCols = 4;
	int bitLength = 32;
	cudaBitMatrix* bitMatrix;
	cudaMallocManaged(&bitMatrix, sizeof(cudaBitMatrix));
	new(bitMatrix) cudaBitMatrix(numRows, numCols, bitLength);

	cudaBitArray* mask;
	cudaMallocManaged(&mask, sizeof(cudaBitArray));
	new(mask) cudaBitArray(bitLength);
	mask->bitwiseInvert(mask);

	cudaBitArray* cache1;
	cudaMallocManaged(&cache1, sizeof(cudaBitArray));
	new(cache1) cudaBitArray(bitLength);

	cudaBitArray* cache2;
	cudaMallocManaged(&cache2, sizeof(cudaBitArray));
	new(cache2) cudaBitArray(bitLength);

	std::cout << "Original BitMatrix:" << std::endl;
	bitMatrix->initMiddleCell();
	bitMatrix->print();

	mask->print();
	// Perform a transition
	bitMatrix->transition(0, 2, 1, 1, mask, 2, cache1, cache2);

	std::cout << "After transition:" << std::endl;
	bitMatrix->print();

	cudaFree(bitMatrix);
	cudaFree(mask);
	cudaFree(cache1);
	cudaFree(cache2);
}

static void testCloneFunctionality() {
	// Test cudaBitArray clone
	cudaBitArray* bitArray = new cudaBitArray(32);
	bitArray->setBit(0, true);
	cudaBitArray* clonedBitArray = new cudaBitArray(32);
	bitArray->clone(clonedBitArray);
	std::cout << "Original BitArray: ";
	bitArray->print();
	std::cout << "Cloned BitArray: ";
	clonedBitArray->print();

	// Test cudaBitVector clone
	cudaBitVector* bitVector = new cudaBitVector(10, 32);
	bitVector->setBit(0, 0, true);
	cudaBitVector* clonedBitVector = new cudaBitVector(10, 32);
	bitVector->clone(clonedBitVector);
	std::cout << "Original BitVector: ";
	bitVector->print();
	std::cout << "Cloned BitVector: ";
	clonedBitVector->print();

	// Test cudaBitMatrix clone
	cudaBitMatrix* bitMatrix = new cudaBitMatrix(4, 4, 32);
	bitMatrix->setBit(0, 0, 0, true);
	cudaBitMatrix* clonedBitMatrix = new cudaBitMatrix(4, 4, 32);
	bitMatrix->clone(clonedBitMatrix);
	std::cout << "Original BitMatrix: ";
	bitMatrix->print();
	std::cout << "Cloned BitMatrix: ";
	clonedBitMatrix->print();

	// Test cudaBatchBitMatrix clone
	cudaBatchBitMatrix* batchBitMatrix = new cudaBatchBitMatrix(2, 4, 4, 32);
	batchBitMatrix->getMatrix(0)->setBit(0, 0, 0, true);
	cudaBatchBitMatrix* clonedBatchBitMatrix = new cudaBatchBitMatrix(2, 4, 4, 32);
	batchBitMatrix->clone(clonedBatchBitMatrix);
	std::cout << "Original BatchBitMatrix: ";
	batchBitMatrix->print();
	std::cout << "Cloned BatchBitMatrix: ";
	clonedBatchBitMatrix->print();

	delete bitArray;
	delete clonedBitArray;
	delete bitVector;
	delete clonedBitVector;
	delete bitMatrix;
	delete clonedBitMatrix;
	delete batchBitMatrix;
	delete clonedBatchBitMatrix;
}

#endif // !CUDABITCLASS_CUH