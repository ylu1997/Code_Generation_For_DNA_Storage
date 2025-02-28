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
		//���������ӵ��������ƺϲ���ע�����ӵ��������������
		mergeLimit(getNumLimit() - 3, getNumLimit() - 2);
		mergeLimit(getNumLimit() - 2, getNumLimit() - 1);
	}

	void mergeLimit(int index1, int index2) {
		// �������ӣ� sequence�ϲ�������ŵ�index1��index2ɾ��
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
		// ��������EditLimit����
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
		// ��limit����2�� ���ԳƵ�pair���ֲ��ߣ����ԳƵķ�ת����ӵ�����
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

	// ������������ÿһ�� pairSequence �� pair ���������ַ�������
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

	// ������������ÿ�� pair �󳤶Ȳ�ľ���ֵ��Ȼ��� pairSequence �е� pair ȡ���ֵ�ٳ��� limit�������� pairSequence ���
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

	// ���һЩ����
	manager.addEditLimit(5, { {"a", "b"}, {"c", "d"} });
	manager.addInstituteLimit(2);
	manager.addDeletionLimit(3);
	manager.addInsertionLimit(3);

	// ��ӡ��ǰ����
	std::cout << "Initial Edit Limits:" << std::endl;
	manager.print();

	// ��Ӳ��ϲ�����
	manager.addEditLimit(2);

	// ��ӡ�ϲ��������
	std::cout << "After Adding and Merging Edit Limits:" << std::endl;
	manager.print();

	manager.mergeLimit(0, 1);
	std::cout << "After Merging Edit Limits:" << std::endl;
	manager.print();

	// ���� exportBitLength ����
	int bitLength = manager.exportBitLength();
	std::cout << "Exported Bit Length: " << bitLength << std::endl;

	// ���� exportMasks ����
	std::vector<std::vector<bool>> masks = manager.exportMasks();
	std::cout << "Exported Masks:" << std::endl;
	for (const auto& mask : masks) {
		for (bool bit : mask) {
			std::cout << bit << " ";
		}
		std::cout << std::endl;
	}

	// ���� exportShiftAmount ����
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


	// ������cudaBitSequence��ȡmask
	cudaBitVector mask = cudaBitVector(masks);
	mask.print();

	// ������cudaEditLimit��ȡgetsequence
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