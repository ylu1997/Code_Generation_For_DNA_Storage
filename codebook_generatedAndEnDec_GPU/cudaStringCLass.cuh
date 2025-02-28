#ifndef CUDASTRINGCLASS_CUH
#define CUDASTRINGCLASS_CUH



#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>

class cudaString {
private:
	char* data;
	int length;
	int effectiveLength;
public:
	cudaString() {
		data = nullptr;
		length = 0;
		effectiveLength = 0;
	}

	cudaString(int n) {
		length = n;
		effectiveLength = 0;
		cudaMallocManaged(&data, length * sizeof(char));
		for (int i = 0; i < length; i++) {
			data[i] = '\0';
		}
	}

	cudaString(const std::string& str) {
		
		length = str.length() + 1;
		effectiveLength = str.length();
		cudaMallocManaged(&data, (length) * sizeof(char));
		for (int i = 0; i < length; i++) {
			if (i < effectiveLength) {
				data[i] = str[i];
			}
			else {
				data[i] = '\0';
			}
		}

		//length = str.length();
		//effectiveLength = length;
		//cudaMallocManaged(&data, (length) * sizeof(char));
		//
		//for (int i = 0; i < length; i++) {
		//	data[i] = str[i];
		//}
	}

	~cudaString() {
		cudaFree(data);
	}

	__host__ __device__  void print() const {
		// 需要打印char的effective长度部分 
		for (int i = 0; i < effectiveLength; i++) {
			printf("%c", data[i]);
		}
		printf("\n");
	}

	__host__ __device__ int len() const {
		return effectiveLength;
	}

	__host__ __device__ void setEmpty() {
		for (int i = 0; i < length; i++) {
			data[i] = '\0';
		}
		effectiveLength = 0;
	}

	__host__ __device__ void setStr(cudaString* str) {
		effectiveLength = str->effectiveLength;
		for (int i = 0; i < length; i++) {
			if (i < str->len())
				data[i] = str->data[i];
			else
				data[i] = '\0';
		}
	}

	__host__ __device__ void getSlice(int start, int end, cudaString* result) {
		if (start >= length || end > length || start > end) {
			result->setEmpty();
		}
		else {
			result->effectiveLength = end - start;
			for (int i = 0; i < result->length; i++) {
				if (i < result->effectiveLength) {
					result->data[i] = data[start + i];
				}
				else {
					result->data[i] = '\0';
				}
			}
		}
	}

	__host__ __device__ bool isEqual(cudaString* str) {
		if (effectiveLength != str->effectiveLength) {
			return false;
		}
		for (int i = 0; i < effectiveLength; i++) {
			if (data[i] != str->data[i]) {
				return false;
			}
		}
		return true;
	}

	__host__ __device__ char getChar(int index) {
		if ((index < effectiveLength)&& (index>=0)) {
			return data[index];
		}
		else {
			return '\0';
		}
	}
	 

};

class cudaPair {
private:
	cudaString* data;
public:
	cudaPair() {
		data = nullptr;
	}

	cudaPair(int n) {
		cudaMallocManaged(&data, 2 * sizeof(cudaString));
		new(data) cudaString(n);
		new(data + 1) cudaString(n);
	}

	cudaPair(const std::string& str1, const std::string& str2) {
		cudaMallocManaged(&data, 2 * sizeof(cudaString));
		new(data) cudaString(str1);
		new(data + 1) cudaString(str2);
	}

	~cudaPair() {
		cudaFree(data);
	}

	__host__ __device__ cudaString* getFirst() {
		return data;
	}

	__host__ __device__ cudaString* getSecond() {
		return data + 1;
	}

	__host__ __device__ cudaString* getPair(int i) {
		return data + i;
	}

	__host__ void print() const {
		data->print();
		(data + 1)->print();
	}
};

class cudaPairSequence {
private:
	cudaPair* data;
	int numPairs;
public:
	cudaPairSequence() {
		data = nullptr;
		numPairs = 0;
	}

	cudaPairSequence(int n, int str_len) {
		numPairs = n;
		cudaMallocManaged(&data, numPairs * sizeof(cudaPair));
		for (int i = 0; i < numPairs; i++) {
			new(data + i) cudaPair(str_len);
		}
	}

	cudaPairSequence(const std::vector<std::pair<std::string, std::string>>& input) {
		numPairs = input.size();
		cudaMallocManaged(&data, numPairs * sizeof(cudaPair));
		for (int i = 0; i < numPairs; i++) {
			new(data + i) cudaPair(input[i].first, input[i].second);
		}
	}

	~cudaPairSequence() {
		for (int i = 0; i < numPairs; i++) {
			data[i].~cudaPair();
		}
		cudaFree(data);
	}

	__host__ __device__ cudaPair* getPair(int i) {
		return data + i;
	}

	__host__ __device__ cudaString* getString(int i, int j) {
		return (data + i)->getPair(j);
	}
	
	__host__ __device__ int getNumPairs() {
		return numPairs;
	}

};

class cudaEditLimit {
public:
	cudaPairSequence* data;
	int numSeqs;
public:
	cudaEditLimit() {
		data = nullptr;
		numSeqs = 0;
	}

	cudaEditLimit(int n, int m, int k) {
		numSeqs = n;
		cudaMallocManaged(&data, numSeqs * sizeof(cudaPairSequence));
		for (int i = 0; i < numSeqs; i++) {
			new(data + i) cudaPairSequence(m, k);
		}
	}

	cudaEditLimit(const std::vector<std::vector<std::pair<std::string, std::string>>>& input) {
		numSeqs = input.size();
		cudaMallocManaged(&data, numSeqs * sizeof(cudaPairSequence));
		for (int i = 0; i < numSeqs; i++) { 
			new(data + i) cudaPairSequence(input[i]);
		}
	}

	~cudaEditLimit() {
		for (int i = 0; i < numSeqs; i++) {
			data[i].~cudaPairSequence();
		}
		cudaFree(data);
	}

	__host__ __device__ cudaPairSequence* getPairSequence(int i) {
		return data + i;
	}

	__host__ __device__ cudaPair* getPair(int i, int j) {
		return (getPairSequence(i))->getPair(j);
	}

	__host__ __device__ cudaString* getString(int i, int j, int k) {
		return (getPairSequence(i))->getString(j, k);
	}


	__host__ void print() const {
		for (int i = 0; i < numSeqs; i++) {
			std::cout << "Sequence " << i << std::endl;
			cudaPairSequence* seq = data + i;
			for (int j = 0; j < seq->getNumPairs(); j++) {
				std::cout << "Pair " << j << std::endl; 
				(data + i)->getPair(j)->print();
			}
		}
	}

	__host__ __device__ int getNumEachSeq(int index) {
		return (data + index)->getNumPairs();
	}

	__host__ __device__ int getSeqNum() {
		return numSeqs;
	}
};

class cudaStringSequence {
private:
	cudaString* data;
	int numStrings;
public:
	cudaStringSequence() {
		data = nullptr;
		numStrings = 0;
	}

	cudaStringSequence(int n, int m) {
		numStrings = n;
		cudaMallocManaged(&data, numStrings * sizeof(cudaString));
		for (int i = 0; i < numStrings; i++) {
			new(data + i) cudaString(m);
		}
	}

	cudaStringSequence(const std::vector<std::string>& sequence) {
		numStrings = sequence.size();
		cudaMallocManaged(&data, numStrings * sizeof(cudaString));
		for (int i = 0; i < numStrings; i++) {
			new(data + i) cudaString(sequence[i]);
		}
	}

	~cudaStringSequence() {
		for (int i = 0; i < numStrings; i++) {
			data[i].~cudaString();
		}
		cudaFree(data);
	}

	__host__ __device__ void print() const {
		for (int i = 0; i < numStrings; i++) {
			//std::cout << "String " << i << std::endl;
			//data[i].print();
			printf("String %d: ", i);
			data[i].print();
		}
	}

	__host__ __device__ void setString(int index, cudaString* s) {
		data[index].setStr(s);
	}

	__host__ __device__ cudaString* getString(int index) {
		return data + index;
	}

	__host__ __device__ int getNumStrings() {
		return numStrings;
	}
};

__global__ void kernelTest(cudaString* str, cudaPair* pair) {
	str->setEmpty();
	pair->getFirst()->setStr(pair->getSecond());
}

__global__ void kernelTestStringSequence(cudaStringSequence* strSeq, cudaString* str) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < strSeq->getNumStrings()) {
		str->getSlice(0, 6, strSeq->getString(idx));
	}
}


static int testStringClass1() {
	std::cout << "Hello, World!" << std::endl;

	// 测试 cudaString
	cudaString* str;
	cudaMallocManaged(&str, sizeof(cudaString));
	new(str) cudaString("Hello, CUDA!");
	std::cout << "cudaString content: ";
	str->print();
	std::cout << "cudaString length: " << str->len() << std::endl;

	// 测试 cudaPair
	cudaPair* pair;
	cudaMallocManaged(&pair, sizeof(cudaPair));
	new(pair) cudaPair("First String", "Second String");
	std::cout << "cudaPair content: " << std::endl;
	pair->print();

	// 调用 kernel 函数
	kernelTest << <1, 1 >> > (str, pair);
	cudaDeviceSynchronize();

	std::cout << "After kernelTest: " << std::endl;
	std::cout << "cudaString content: ";
	str->print();
	std::cout << "cudaPair content: " << std::endl;
	pair->print();

	cudaFree(str);
	cudaFree(pair);


	cudaStringSequence* strSeq;
	cudaMallocManaged(&strSeq, sizeof(cudaStringSequence));
	new(strSeq) cudaStringSequence(3, 25);
	// 调用 kernel 函数测试 cudaStringSequence
	cudaString* newStr;
	cudaMallocManaged(&newStr, sizeof(cudaString));
	new(newStr) cudaString("New String");
	kernelTestStringSequence << <1, 3 >> > (strSeq, newStr);
	cudaDeviceSynchronize();

	std::cout << "After kernelTestStringSequence: " << std::endl;
	strSeq->print();


	cudaFree(strSeq);
	return 0;
}

#endif // !CUDASTRINGCLASS_CUH