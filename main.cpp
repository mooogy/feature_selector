#include <iostream>
#include <string>

#include <vector>

#include <limits>

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
		fmt::print("Please enter the filename / path of the dataset you would like to analyze.\n> ");
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
			fmt::print("ERROR: Could not find file! Please try again.\n\n");
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

	fmt::print("\tUsing feature(s): {} | Accuracy: {:.2f}%\n", feature_set, acc);

	return acc;
}

[[nodiscard]] float default_rate(const Dataset& dataset) {
	Label majority = (dataset.label_one_count_ > dataset.label_two_count_) ? Label::One : Label::Two;
	
	if (majority == Label::One) {
		return (float)dataset.label_one_count_ / dataset.data_.size() * 100;
	} else {
		return (float)dataset.label_two_count_ / dataset.data_.size() * 100;
	}
}

// SEARCH ALGO CODE

void forward_selection(const Dataset& dataset) {
	fmt::print("\nBeginning Forward Selection...\n");

	auto start = std::chrono::high_resolution_clock::now();

	std::vector<int> possible_feature_set;
	for (int i = 1; i <= dataset.features_; i++) possible_feature_set.push_back(i);

	std::vector<int> selected_feature_set;
	float overall_best_acc = std::numeric_limits<float>::min();
	std::vector<int> overall_best_feature_set;

	while (true) {
		if (selected_feature_set.size() == possible_feature_set.size()) break;

		int depth = selected_feature_set.size() + 1;
		float level_best_acc = std::numeric_limits<float>::min();
		std::vector<int> level_best_feature_set;

		fmt::print("\nSearching through depth: {}\n", depth);

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

void backwards_elimination(const Dataset& dataset) {
	fmt::print("Beginning backwards elimination...\n");
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<int> selected_feature_set;
	for (int i = 1; i <= dataset.features_; i++) selected_feature_set.push_back(i);


	int depth = 1;
	float overall_best_acc = std::numeric_limits<float>::min();
	std::vector<int> overall_best_feature_set = selected_feature_set;

	while (true) {
		if (selected_feature_set.size() == 1) break;

		float level_best_acc = std::numeric_limits<float>::min();
		int level_best_removed = 0;

		fmt::print("\nSearching through depth: {}\n", depth);
		for (int feature_index : selected_feature_set) {
			std::vector<int> feature_set;
			for (int f : selected_feature_set)
				if (f != feature_index) feature_set.push_back(f);

			float acc = leave_one_out_validation(dataset.data_, feature_set);
			if (acc > level_best_acc) {
				level_best_acc = acc;
				level_best_removed = feature_index;
			}
		}


		selected_feature_set.erase(std::find(selected_feature_set.begin(), selected_feature_set.end(), level_best_removed));

		fmt::print("Feature subset {} was best at depth {}. | Accuracy: {:.2f}%\n", selected_feature_set, depth, level_best_acc);
		depth += 1;

		if (level_best_acc > overall_best_acc) {
			overall_best_acc = level_best_acc;
			overall_best_feature_set = selected_feature_set;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();

	fmt::print("\nBackwards elimination complete!\n");
	fmt::print("Feature subset {} is best overall with an accuracy of {:.2f}%\n", overall_best_feature_set, overall_best_acc);

	std::chrono::duration<float> elapsed = end - start;
	fmt::print("Backwards Elimination Time: {:.2f}s\n", elapsed.count());
}

// MAIN PROGRAM CODE

int main() {
	fmt::print("Welcome to the Feature Selection Program!\n");
	const Dataset dataset = read_dataset();

	fmt::print("\n== DATASET STATS ==\n");
	fmt::print("FEATURES: {} | LABEL ONE COUNT: {} | LABEL TWO COUNT: {}\n",
			dataset.features_, dataset.label_one_count_, dataset.label_two_count_);

	fmt::print("Please choose a feature selection algorithm.\n\t1) Forward Selection\n\t2) Backwards Elimination\n");


	while (true) {
		fmt::print("(1/2): ");
		std::string algo_choice = "";
		std::cin >> algo_choice;


		fmt::print("\nNearest Neighbor with no features. (default_rate) | Acc: {:.2f}%\n", default_rate(dataset));

		if (algo_choice == "1") {
			forward_selection(dataset);
			break;
		}

		if (algo_choice == "2") {
			backwards_elimination(dataset);
			break;
		}
		fmt::print("ERROR: Invalid algorithm choice. Please enter 1 or 2.\n");
	}

	return 0;
}
