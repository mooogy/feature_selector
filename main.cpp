#include <iostream>
#include <string>

#include <vector>

#include <limits>
#include <cmath>

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

float euclidean_distance(const Record& first, const Record& second, const std::vector<int>& feature_set) {
	float total_square_diff = 0;
	
	// index is feature - 1
	for (const int& feature : feature_set) {
		float diff = first.features_[feature - 1] - second.features_[feature - 1];
		total_square_diff += diff * diff; 
	}

	return std::sqrt(total_square_diff);
}

Label nn(const std::vector<Record>& data, const int unlabeled_index, const std::vector<int>& feature_set) {
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

float leave_one_out_validation(const std::vector<Record>& data, const std::vector<int>& feature_set) {
	float record_count = data.size();
	float total_right = 0;

	for (int i = 0; i < record_count; i++) {
		if (data[i].label_ == nn(data, i, feature_set)) total_right += 1;
	}

	float acc = (total_right / record_count) * 100;

	fmt::print("\tUsing feature(s): {0} | Accuracy: {1:.2f}\n", feature_set, acc);

	return acc;
}



// MAIN PROGRAM CODE

int main() {
	auto dataset = read_dataset();

	fmt::print("\n== DATASET STATS ==\n");
	fmt::print("LABEL ONE COUNT: {0} | LABEL TWO COUNT: {1}\n",
			dataset.label_one_count_, dataset.label_two_count_);

	fmt::print("RECORD ONE: {}\n", dataset.data_[0].features_);
	
	std::vector<int> f_set = {10, 8, 2};
	float acc = leave_one_out_validation(dataset.data_, f_set);
	
	fmt::print("ACC: {:.2f}", acc);

	return 0;
}
