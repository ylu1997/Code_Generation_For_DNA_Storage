#ifndef CUDAEDITLIMITCLASS_CUH
#define CUDAEDITLIMITCLASS_CUH

#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<vector>
#include <numeric>
#include "cudaStringClass.cuh"
#include "cudaBitClass.cuh"

using EditPair = std::pair<std::string, std::string>;
using EditPairSeq = std::vector<EditPair>;
using EditLimit = std::pair<int, EditPairSeq>;
using EditLimitList = std::vector<EditLimit>;

class EditLimitManager {
private:
	EditLimitList editLimitList = EditLimitList();
public:
	EditLimitManager() {
	}

	void addEditLimit(int limit, const EditPairSeq& editPairSeq) {
		editLimitList.push_back(EditLimit(limit, editPairSeq));
	}

	void addInstituteLimit(int limit) {
		addSingleLimitPair(limit, "*", "*");
	}

	void addDeletionLimit(int limit) {
		addSingleLimitPair(limit, "*", "");
	}

	void addInsertionLimit(int limit) {
		addSingleLimitPair(limit, "", "*");
	}

	void addEditLimit(int limit) {
		addDeletionLimit(0);
		addInsertionLimit(0);
		addInstituteLimit(limit);
		//将上述增加的三个限制合并，注意增加的是最后三个限制
		mergeLimit(getNumLimit() - 3, getNumLimit() - 2);
		mergeLimit(getNumLimit() - 2, getNumLimit() - 1);
	}

	void mergeLimit(int index1, int index2) {
		// 限制增加， sequence合并，结果放到index1，index2删除
		if (index1 >= 0 && index1 < editLimitList.size() && index2 >= 0 && index2 < editLimitList.size()) {
			EditLimit& limit1 = editLimitList[index1];
			EditLimit& limit2 = editLimitList[index2];
			limit1.first += limit2.first;
			limit1.second.insert(limit1.second.end(), limit2.second.begin(), limit2.second.end());
			editLimitList.erase(editLimitList.begin() + index2);
		}
		else {
			std::cerr << "Index out of range" << std::endl;
		}
	}

	void addSingleLimitPair(int limit, std::string s1, std::string s2) {
		EditPair pair = EditPair(s1, s2);
		addEditLimit(limit, EditPairSeq(1, pair));
	}

	int getNumLimit() {
		// 返回所有EditLimit个数
		return editLimitList.size();
	}

	void print() {
		for (int i = 0; i < editLimitList.size(); i++) {
			std::cout << "Limit: " << editLimitList[i].first << std::endl;
			for (int j = 0; j < editLimitList[i].second.size(); j++) {
				std::cout << "Pair: " << editLimitList[i].second[j].first << " " << editLimitList[i].second[j].second << std::endl;
			}
		}
	}

	int exportBitLength() const {
		return std::accumulate(editLimitList.begin(), editLimitList.end(), 1, [](int product, const EditLimit& element) {
			return product * (element.first + 1);
			});
	}

	std::vector<int> index2tuple(int index, const std::vector<int>& basis) const {
		std::vector<int> ans;
		for (auto it = basis.rbegin(); it != basis.rend(); ++it) {
			ans.push_back(index % *it);
			index = index / *it;
		}
		std::reverse(ans.begin(), ans.end());
		return ans;
	}

	std::vector<std::vector<bool>> exportMasks() {
		std::vector<int> upper_limit;
		for (const auto& element : editLimitList) {
			upper_limit.push_back(element.first + 1);
		}
		std::vector<std::vector<bool>> masks(editLimitList.size(), std::vector<bool>(exportBitLength(), false));
		for (int i = 0; i < exportBitLength(); ++i) {
			std::vector<int> tup = index2tuple(i, upper_limit);
			for (size_t j = 0; j < editLimitList.size(); ++j) {
				if (tup[j] < upper_limit[j] - 1) {
					masks[j][i] = true;
				}
			}
		}
		return masks;
	}

	std::vector<int> exportShiftAmount() {
		std::vector<int> upper_limit;
		for (const auto& element : editLimitList) {
			upper_limit.push_back(element.first + 1);
		}
		std::vector<int> shiftamount(upper_limit.size(), 1);
		for (size_t i = 1; i < upper_limit.size(); ++i) {
			shiftamount[i] = shiftamount[i - 1] * upper_limit[upper_limit.size() - i];
		}
		std::reverse(shiftamount.begin(), shiftamount.end());
		return shiftamount;
	};


	std::vector<std::vector<std::pair<std::string, std::string>>> exportSequence() {
		std::vector<std::vector<std::pair<std::string, std::string>>> ans;
		for (const auto& element : editLimitList) {
			std::vector<std::pair<std::string, std::string>> temp;
			for (const auto& pair : element.second) {
				temp.push_back(pair);
			}
			ans.push_back(temp);
		}
		return ans;
	}

	void symmetric() {
		// 将limit乘以2， 将对称的pair保持不边，不对称的反转后添加到后面
		for (auto& element : editLimitList) {
			element.first *= 2;
			EditPairSeq new_seq;
			for (const auto& pair : element.second) {
				new_seq.push_back(pair);
				if (pair.first != pair.second) {
					new_seq.push_back({ pair.second, pair.first });
				}
			}
			element.second = new_seq;
		}
	}

	// 新增方法：对每一个 pairSequence 的 pair 返回最大的字符串长度
	int getMaxStringLength() const {
		int maxLength = 0;
		for (const auto& editLimit : editLimitList) {
			for (const auto& pair : editLimit.second) {
				int length1 = pair.first.length();
				int length2 = pair.second.length();
				maxLength = std::max(maxLength, std::max(length1, length2));
			}
		}
		return maxLength;
	}

	// 新增方法：对每个 pair 求长度差的绝对值，然后对 pairSequence 中的 pair 取最大值再乘以 limit，对所有 pairSequence 求和
	int getMaxLengthDifferenceSum() const {
		int totalSum = 0;
		for (const auto& editLimit : editLimitList) {
			int maxDifference = 0;
			for (const auto& pair : editLimit.second) {
				int length1 = pair.first.length();
				int length2 = pair.second.length();
				int difference = std::abs(length1 - length2);
				maxDifference = std::max(maxDifference, difference);
			}
			totalSum += maxDifference * editLimit.first;
		}
		return totalSum;
	}

};


void testEditLimitManager() {
	EditLimitManager manager;

	// 添加一些限制
	manager.addEditLimit(5, { {"a", "b"}, {"c", "d"} });
	manager.addInstituteLimit(2);
	manager.addDeletionLimit(3);
	manager.addInsertionLimit(3);

	// 打印当前限制
	std::cout << "Initial Edit Limits:" << std::endl;
	manager.print();

	// 添加并合并限制
	manager.addEditLimit(2);

	// 打印合并后的限制
	std::cout << "After Adding and Merging Edit Limits:" << std::endl;
	manager.print();

	manager.mergeLimit(0, 1);
	std::cout << "After Merging Edit Limits:" << std::endl;
	manager.print();

	// 测试 exportBitLength 方法
	int bitLength = manager.exportBitLength();
	std::cout << "Exported Bit Length: " << bitLength << std::endl;

	// 测试 exportMasks 方法
	std::vector<std::vector<bool>> masks = manager.exportMasks();
	std::cout << "Exported Masks:" << std::endl;
	for (const auto& mask : masks) {
		for (bool bit : mask) {
			std::cout << bit << " ";
		}
		std::cout << std::endl;
	}

	// 测试 exportShiftAmount 方法
	std::vector<int> shiftAmounts = manager.exportShiftAmount();
	std::cout << "Exported Shift Amounts:" << std::endl;
	for (int amount : shiftAmounts) {
		std::cout << amount << " ";
	}
	std::cout << std::endl;

	std::cout << "Exported Sequences:" << std::endl;
	std::vector<std::vector<std::pair<std::string, std::string>>> sequences = manager.exportSequence();
	for (const auto& sequence : sequences) {
		std::cout << "Sequence:" << std::endl;
		for (const auto& pair : sequence) {
			std::cout << pair.first << " " << pair.second << std::endl;
		}
	}

	std::cout << "Symmetric Edit Limits:" << std::endl;
	manager.symmetric();
	manager.print();


	// 测试用cudaBitSequence获取mask
	cudaBitVector mask = cudaBitVector(masks);
	mask.print();

	// 测试用cudaEditLimit获取getsequence
	// show sequence
	std::cout << "Show sequence" << std::endl;
	for (const auto& sequence : sequences) {
		std::cout << "Sequence:" << std::endl;
		for (const auto& pair : sequence) {
			std::cout << pair.first << " " << pair.second << std::endl;
		}
	}
	std::vector<std::vector<std::pair<std::string, std::string>>> sequences2 = { { { "*", "*"} } };
	cudaEditLimit editLimit = cudaEditLimit(sequences2);
	editLimit.print();

}

#endif // !CUDAEDITLIMITCLASS_CUH