#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>
#include <numeric>
#include"bitOperation.cuh"
#include "cudaStringClass.cuh"
#include "BatchBitMatrix.cuh"
#include "AssistantData.cuh"
//#include "cudaBitClass.cuh"
#include "EditLimitListManager.cuh"
//#include "cudaIntVector.cuh"
//#include "cudaReachability.cuh"
 



__device__ int index_to_row(int index1, int index2, int rowStart, int rowNum) {
	return ((index1 + index2 + abs(index2 - index1)) / 2 + rowStart) % rowNum;
}

__device__ int index_to_col(int index1, int index2, int colStart) {
	return (index2 - index1 + colStart);
}

__device__ bool isCase(cudaString* s1, cudaString* s2, cudaString* p1, cudaString* p2) {
	if (s1->len() == p1->len() && s2->len() == p2->len()) {
		for (int i = 0; i < s1->len(); i++) {
			char c_p1 = p1->getChar(i);
			char c_s1 = s1->getChar(i);
			if (c_p1 != '*') {
				if (c_p1 == '=') {
					if (i < s2->len()) {
						if (c_s1 != s2->getChar(i)) {
							return false;
						}
					}
					else {
						return false;
					}
				}
				else {
					if (c_p1 != c_s1) {
						return false;
					}
				}
			}
		}
		for (int j = 0; j < s2->len(); j++) {
			char c_p2 = p2->getChar(j);
			char c_s2 = s2->getChar(j);
			if (c_p2 != '*') {
				if (c_p2 == '=') {
					if (j < s1->len()) {
						if (c_s2 != s1->getChar(j)) {
							return false;
						}
					}
					else {
						return false;
					}
				}
				else {
					if (c_p2 != c_s2) {
						return false;
					}
				}
			}
		}
		return true;
	}
	else {
		return false;
	}
}


__host__ __device__ int py_div(int a, int b) {
	int q = a / b;
	// 如果余数不为0且a和b符号不同，结果减1（模拟Python地板除）
	if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
	return q;
}

__host__ __device__ int py_mod(int a, int b) {
	int r = a % b;
	// 如果余数为负，调整为和除数同符号
	if (r < 0) r += (b < 0 ? -b : b);  // 代替 abs(b)
	return r;
}

__device__ void mat_update(d_BatchBitMatrix* bMats,int batchIndex,
	cudaString* s1, cudaString* s2,
	cudaString* p1, cudaString* p2, 
	int index1, int index2, d_AssistantArray* assArr, int maskIndex, int rowStart, int colStart ) {
	int rowTarget, colTarget;
	rowTarget = index_to_row(index1, index2, rowStart, bMats->d_getNumRows());
	colTarget = index_to_col(index1, index2, colStart);
	int rowSource, colSource;
	rowSource = index_to_row(index1 - p1->len(), index2 - p2-> len(), rowStart, bMats->d_getNumRows());
	colSource = index_to_col(index1 - p1->len(), index2 - p2->len(), colStart);

	setEmpty(assArr->getCache1(batchIndex), bMats->d_getNumUInt());
	setEmpty(assArr->getCache2(batchIndex), bMats->d_getNumUInt());
	if (isCase(s1, s2, p1, p2) && (abs(index1 - index2 - p1->len() + p2->len()) <= bMats->d_getNumCols() / 2)) {
		bMats->d_transition(batchIndex, rowSource, colSource, rowTarget, colTarget, assArr->getMask(maskIndex), assArr->getShiftAmount(maskIndex), assArr->getCache1(maskIndex), assArr->getCache2(maskIndex));
	}
}

__device__ void kernelReachability(cudaEditLimit* editLimit, cudaString* s1, cudaString* s2, int batchIndex, int startIndex, int endIndex, d_BatchBitMatrix* bMats, d_AssistantArray* assArr, int startRow, int* nextRow, cudaString* subs1, cudaString* subs2, cudaString* anyChar) {
	int wid = bMats->d_getNumCols() / 2;
	int dp = bMats->d_getNumRows() - 1;
	int d;
	int index1, index2;
	int row, col;
	for (int i = startIndex; i < endIndex; i++) {
		d = min(i, wid);
		for (int j = -2 * d; j < 1; j++) {
			index1 = (i + py_div(j, 2) * py_mod(j, 2));
			index2 = (i + py_div(j, 2) * py_mod(j + 1, 2));
			s1->getSlice(index1, index1 + 1, subs1);
			s2->getSlice(index2, index2 + 1, subs2);
			col = index_to_col(index1, index2, wid);
			row = index_to_row(index1, index2, 1 - startIndex + startRow, bMats->d_getNumRows());
			setEmpty(bMats->d_getCellPointer(batchIndex, row, col), bMats->d_getNumUInt());
			if (subs1->isEqual(subs2)) {
				mat_update(bMats, batchIndex, subs1, subs2, anyChar, anyChar, index1, index2, assArr, -1, 1 + startRow - startIndex, wid);
			}
			else {
				for (int i_th = 0; i_th < editLimit->getSeqNum(); i_th++) {
					for (int j_th = 0; j_th < editLimit->getNumEachSeq(i_th); j_th++) {
						cudaString* p1 = editLimit->getString(i_th, j_th, 0);
						cudaString* p2 = editLimit->getString(i_th, j_th, 1);
						s1->getSlice(max(0, index1 - p1->len() + 1), index1 + 1, subs1);
						s2->getSlice(max(0, index2 - p2->len() + 1), index2 + 1, subs2);
						mat_update(bMats, batchIndex, subs1, subs2, p1, p2, index1, index2, assArr, i_th, 1 + startRow - startIndex, wid);
					}
				}
			}

		}
	}
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
		*nextRow = (startRow + endIndex - startIndex) % (dp + 1);
	}
}
 
__device__ void kernel_AugumentReachability(cudaEditLimit* editLimit, cudaString* s1, cudaString* s2, int batchIndex, int augAmount, d_BatchBitMatrix* bMats, d_AssistantArray* assArr, int startRow, int* nextRow, cudaString* subs1, cudaString* subs2, cudaString* anyChar) {
	kernelReachability(editLimit, s1, s2, batchIndex, s1->len() - augAmount, s1->len(), bMats, assArr, startRow, nextRow, subs1, subs2, anyChar);
}


__global__ void testKernel(cudaStringSequence* seuqnece, d_BatchBitMatrix* bMats, cudaEditLimit* editLimit, d_AssistantArray* assArr, cudaStringSequence * substr1, cudaStringSequence* substr2, int startRow , int* nextRow, int augAmount, cudaString* anyChar) {
	int t_id = threadIdx.x + blockIdx.x * blockDim.x;
	printf("t_id: %d\n", t_id);
	if (t_id < bMats->d_getNumBatch())
	{
		cudaString* s1 = seuqnece->getString(t_id * 2);
		cudaString* s2 = seuqnece->getString(t_id * 2 + 1);
		cudaString* subs1 = substr1->getString(t_id);
		cudaString* subs2 = substr2->getString(t_id); 
		//kernel_AugumentReachability(editLimit, s1, s2, t_id, augAmount, bMats, assArr, startRow, nextRow, subs1, subs2, anyChar);
	}
}

#include <random>
std::vector<std::string> generateStringSequence(int n, int m) {
	std::vector<std::string> sequence;
	std::string chars = "AGCT";
	std::default_random_engine rng(std::random_device{}());
	std::uniform_int_distribution<> dist(0, chars.size() - 1);

	for (int i = 0; i < n; ++i) {
		std::string str;
		for (int j = 0; j < m; ++j) {
			str += chars[dist(rng)];
		}
		sequence.push_back(str);
	}
	return sequence;
}

class ReachabilityManager {
private:
	BatchBitMatrix* bMats=nullptr;
	AssistantArray* assArr=nullptr;
	cudaStringSequence* stringSequence = nullptr;
	cudaStringSequence* substr1 = nullptr;
	cudaStringSequence* substr2 = nullptr;
	cudaString* anyChar = nullptr;
	cudaEditLimit* cuEditLimit = nullptr;
public:
	EditLimitManager editLimitManager;

	ReachabilityManager() {
		cudaMallocManaged(&anyChar, sizeof(cudaString));
		new(anyChar) cudaString("*");
		std::cout << "reachabilityCalculator constructed" << std::endl;
	}

	~ReachabilityManager() {
		// 先判断是否为 null 再释放
		if (anyChar != NULL) {
			cudaFree(anyChar);
		}
		if (stringSequence != NULL) {
			cudaFree(stringSequence);
		}
		if (substr1 != NULL) {
			cudaFree(substr1);
		}
		if (substr2 != NULL) {
			cudaFree(substr2);
		}
		if (cuEditLimit != NULL) {
			cudaFree(cuEditLimit);
		}
		if (bMats != NULL) {
			delete bMats;
		}
		if (assArr != NULL) {
			delete assArr;
		}

		std::cout << "reachabilityCalculator destructed" << std::endl;
	}

	int bitLength() {
		return editLimitManager.exportBitLength();
	}

	int maxLengthSeq() {
		return editLimitManager.getMaxStringLength();
	}

	int widthWindow() {
		return editLimitManager.getMaxLengthDifferenceSum();
	}

	int depthWindow() {
		return editLimitManager.getMaxStringLength();
	}

	void setSequence(const std::vector<std::string>& sequence) {
		if (stringSequence != nullptr) {
			cudaFree(stringSequence);
		}
		cudaMallocManaged(&stringSequence, sizeof(cudaStringSequence));
		new(stringSequence) cudaStringSequence(sequence);
	}

	void establishEditLimit() {
		cudaMallocManaged(&cuEditLimit, sizeof(cudaEditLimit));
		new(cuEditLimit) cudaEditLimit(editLimitManager.exportSequence());
	}

	void loadBitData(int batchNum, int matRow, int matCol) {
		bMats = new BatchBitMatrix(batchNum, matRow, matCol, editLimitManager.exportBitLength());
		assArr = new AssistantArray(batchNum, batchNum, editLimitManager.exportBitLength(), editLimitManager.exportMasks(), editLimitManager.exportShiftAmount());
	}

	void printMatrix() {
		std::cout << "printMatrix: " << std::endl;
		bMats->h_mat->h_printBatchMatrix();
	}

	void printString() {
		std::cout << "printString: " << std::endl;
		stringSequence->print();
	}

	void printEditLimit() {
		std::cout << "printEditLimit: " << std::endl;
		cuEditLimit->print();
	}

	void printMasks() {
		std::cout << "printMasks: " << std::endl;
		assArr->h_array->printMasks();
	}

	void printShiftAmounts() {
		std::cout << "printShiftAmounts: " << std::endl;
		assArr->h_array->printShiftAmounts();
	}	

	void test() {
		// 先生成序列
		int batchNum = 20000;
		int seqlen = 100;
		std::vector<std::string> sequence = generateStringSequence(2 * batchNum, seqlen); 
		std::cout << "after generated" << std::endl;
		setSequence(sequence);
		std::cout << "after set sequence" << std::endl;
		// 打印序列  
		// 生成编辑限制
		editLimitManager.addEditLimit(2);
		std::cout << "Loading Data" << std::endl;
		loadBitData(batchNum, widthWindow() * 2 + 1, depthWindow() + 1);
		std::cout << "finish loading data" << std::endl;
		int* nextRow = nullptr;
		cudaMallocManaged(&nextRow, sizeof(int));
		nextRow[0] = 0;
		testKernel << <1, 1 >> > (stringSequence, bMats->d_mat, cuEditLimit, assArr->d_array, substr1, substr2, 0, nextRow, seqlen, anyChar);
		cudaDeviceSynchronize(); 
		cudaFree(nextRow);

	}
};


 
#include <windows.h>
int main() { 
	//测试
	ReachabilityManager reachabilityManager;
	reachabilityManager.test();	
	std::cout << "main end" << std::endl;
	return 0;
}
 
