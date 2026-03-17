#ifndef NN_H
#define NN_H

#include <vector>
#include <limits>

#include "data.h"

[[nodiscard]] float euclidean_distance(const Record& first, const Record& second, const std::set<int>& features) {
	float total_square_diff = 0;

	// index is feature - 1
	for (const int& feature : features) {
		float diff = first.features_[feature - 1] - second.features_[feature - 1];
		total_square_diff += diff * diff; 
	}

	return std::sqrt(total_square_diff);
}

[[nodiscard]] Label nn(const std::vector<Record>& data, const int unlabeled_index, const std::set<int>& features) {
	const Record& unlabeled = data[unlabeled_index];

	Label nearest_label = Label::One;
	float nearest_distance = std::numeric_limits<float>::max();

	for (int i = 0; i < data.size(); i++) {
		if (i == unlabeled_index) continue;

		float distance = euclidean_distance(data[i], unlabeled, features);

		if (distance < nearest_distance) {
			nearest_label = data[i].label_;
			nearest_distance = distance;
		}
	}

	return nearest_label;
}

[[nodiscard]] float leave_one_out_validation(const std::vector<Record>& data, const std::set<int>& features) {
	float record_count = data.size();
	float total_right = 0;

	for (int i = 0; i < record_count; i++) {
		if (data[i].label_ == nn(data, i, features)) total_right += 1;
	}

	float acc = (total_right / record_count) * 100;
	return acc;
}

#endif
