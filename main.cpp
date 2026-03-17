#include <iostream>
#include <string>

#include <vector>

#include <limits>
#include <numeric>

#include <chrono>

// NON C++ STDLIB
// https://github.com/vincentlaucsb/csv-parser
#include "csv.hpp"

// https://fmt.dev/12.0/
#include <fmt/base.h>
#include <fmt/ranges.h>

enum class Label : char {
	One = 1,
	Two = 2
};

struct Record {
	Record(Label label, std::vector<float> features) : label_(label), features_(std::move(features)) {}

	Label label_;
	std::vector<float> features_;
};

struct Dataset {
	Dataset(std::vector<Record> data, int features, int label_one, int label_two) :
		data_(std::move(data)), features_(features),
		label_one_count_(label_one), label_two_count_(label_two) {}

	std::vector<Record> data_;
	int features_;
	int label_one_count_;
	int label_two_count_;
};

// READING DATASET CODE

[[nodiscard]] Dataset read_dataset() {
	csv::CSVFormat format;
	format.delimiter(' ').no_header();

	std::vector<Record> data;
	int label_one_count = 0;
	int label_two_count = 0;

	while (true) {
		std::string filename;
		std::cin >> filename;

		try {
			csv::CSVReader reader(filename, format);

			for (csv::CSVRow& row: reader) {

				Label label;
				std::vector<float> features;
				int cols_read = 0;

				for (csv::CSVField& field: row) {
					auto text = field.get<>();
					if (text == "") continue;

					float value = std::stof(text);
					cols_read += 1;

					if (cols_read == 1) {
						label = (value == 1.0) ? Label::One : Label::Two;

						if (label == Label::One) {
							label_one_count++;
						} else {
							label_two_count++;
						}

						continue;
					}

					features.push_back(value);
				}

				data.emplace_back(label, features);
			}
			break;
		} catch (...) {
			std::cout << "ERROR: Could not find file! Please try again.\n";
		}
	}

	return Dataset(data, data[0].features_.size(), label_one_count, label_two_count);
}

// NN AND VALIDATION CODE

[[nodiscard]] float euclidean_distance(const Record& first, const Record& second, const std::vector<int>& feature_set) {
	float total_square_diff = 0;

	// index is feature - 1
	for (const int& feature : feature_set) {
		float diff = first.features_[feature - 1] - second.features_[feature - 1];
		total_square_diff += diff * diff; 
	}

	return std::sqrt(total_square_diff);
}

[[nodiscard]] Label nn(const std::vector<Record>& data, const int unlabeled_index, const std::vector<int>& feature_set) {
	const Record& unlabeled = data[unlabeled_index];

	Label nearest_label = Label::One;
	float nearest_distance = std::numeric_limits<float>::max();

	for (int i = 0; i < data.size(); i++) {
		if (i == unlabeled_index) continue;

		float distance = euclidean_distance(data[i], unlabeled, feature_set);

		if (distance < nearest_distance) {
			nearest_label = data[i].label_;
			nearest_distance = distance;
		}
	}

	return nearest_label;
}

[[nodiscard]] float leave_one_out_validation(const std::vector<Record>& data, const std::vector<int>& feature_set) {
	float record_count = data.size();
	float total_right = 0;

	for (int i = 0; i < record_count; i++) {
		if (data[i].label_ == nn(data, i, feature_set)) total_right += 1;
	}

	float acc = (total_right / record_count) * 100;

	fmt::print("\tUsing feature(s): {0} | Accuracy: {1:.2f}\n", feature_set, acc);

	return acc;
}

// SEARCH ALGO CODE

void forward_selection(const Dataset& dataset) {
	fmt::print("\nBeginning Forward Selection...\n");

	auto start = std::chrono::high_resolution_clock::now();

	std::vector<int> possible_feature_set;
	std::iota(possible_feature_set.begin(), possible_feature_set.end(), 1);
	for (int i = 1; i <= dataset.features_; i++) possible_feature_set.push_back(i);

	std::vector<int> selected_feature_set;
	float overall_best_acc = std::numeric_limits<float>::min();
	std::vector<int> overall_best_feature_set;

	while (true) {
		if (selected_feature_set.size() == possible_feature_set.size()) break;

		int depth = selected_feature_set.size() + 1;
		float level_best_acc = std::numeric_limits<float>::min();
		std::vector<int> level_best_feature_set;

		fmt::print("Searching through depth: {}\n", depth);

		for (int feature_index : possible_feature_set) {
			// skip already selected features
			if (std::find(selected_feature_set.begin(), selected_feature_set.end(), feature_index) != selected_feature_set.end()) continue;

			std::vector<int> feature_set = selected_feature_set;
			feature_set.push_back(feature_index);

			float acc = leave_one_out_validation(dataset.data_, feature_set);
			if (acc > level_best_acc) {
				level_best_acc = acc;
				level_best_feature_set = feature_set;
			}
		}

		fmt::print("Feature subset {} was best at depth {}. | Accuracy: {:.2f}%\n", level_best_feature_set, depth, level_best_acc);

		if (level_best_acc > overall_best_acc) {
			overall_best_acc = level_best_acc;
			overall_best_feature_set = level_best_feature_set;
		}

		selected_feature_set = level_best_feature_set;
	}

	auto end = std::chrono::high_resolution_clock::now();

	fmt::print("\nSearch Complete!\n");
	fmt::print("Feature subset {} is best overall with an accuracy of {:.2f}%\n", overall_best_feature_set, overall_best_acc);

	std::chrono::duration<float> elapsed = end - start;
	fmt::print("Forward Selection Time Taken: {:.2f}s\n", elapsed.count());
}


// MAIN PROGRAM CODE

int main() {
	const Dataset dataset = read_dataset();

	fmt::print("\n== DATASET STATS ==\n");
	fmt::print("LABEL ONE COUNT: {0} | LABEL TWO COUNT: {1}\n",
			dataset.label_one_count_, dataset.label_two_count_);

	forward_selection(dataset);

	return 0;
}
