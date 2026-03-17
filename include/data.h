#ifndef DATA_H
#define DATA_H

// CPP STDLIB

#include <iostream>
#include <vector>
#include <utility>
#include <string>

// UNORIGINAL SUBROUTINE

#include <fmt/base.h>
#include "csv.hpp"

enum class Label : char {
	One = 1,
	Two = 2
};

struct Record {
	Record(Label label, std::vector<float> features) 
		: label_(label), features_(std::move(features)) {}

	Label label_;
	std::vector<float> features_;
};

struct Dataset {
	Dataset(std::vector<Record> data, int features, int label_one, int label_two) :
		data_(std::move(data)), num_features_(features),
		label_one_count_(label_one), label_two_count_(label_two) {}

	std::vector<Record> data_;
	int num_features_;
	int label_one_count_;
	int label_two_count_;
};

[[nodiscard]] float default_rate(const Dataset& dataset) {
	Label majority = (dataset.label_one_count_ > dataset.label_two_count_) ? Label::One : Label::Two;
	
	if (majority == Label::One) {
		return (float)dataset.label_one_count_ / dataset.data_.size() * 100;
	} else {
		return (float)dataset.label_two_count_ / dataset.data_.size() * 100;
	}
}

[[nodiscard]] Dataset read_dataset() {
	// format for the text files
	csv::CSVFormat format;
	format.delimiter(' ').no_header();

	// dataset stats
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

				Label label = Label::One;
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
						} 
						else {
							label_two_count++;
						}

						continue;
					}

					features.push_back(value);
				}

				data.emplace_back(label, features);
			}
			break;
		}
		catch (...) {
			fmt::print("ERROR: Could not find file! Please try again.\n\n");
		}
	}

	return Dataset(data, data[0].features_.size(), label_one_count, label_two_count);
}

#endif
