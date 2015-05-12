#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


// HYQ: adapted from accuracy_layer.cpp
template <typename Dtype>
PerClassAccuracyLayer<Dtype>::PerClassAccuracyLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {
  CHECK(param.has_per_class_accuracy_param()) 
      << "PerClassAccuracyLayer requires a per_class_accuracy_param";
  const string& c_file = param.per_class_accuracy_param().classifier_info_file();
  LOG(INFO) << "Opening file " << c_file;
  std::ifstream infile(c_file.c_str());

  string classifier_name;
  int num_classes;

  int class_label;
  string class_name;
  float class_prior;
  float class_lrmult;

  infile >> classifier_name;
  infile >> num_classes;

  classifier_name_ = classifier_name;
  num_classes_     = num_classes;

  // get information for each class, and dynamically declare one top blob per
  // class
  for (int i=0;i<num_classes;i++){
    infile >> class_label;
    infile >> class_name;
    infile >> class_prior;
    infile >> class_lrmult;
    class_labels_.push_back (class_label);
    class_names_.push_back  (class_name);
    class_priors_.push_back (class_prior);
    class_lrmults_.push_back(class_lrmult);
    class_TPs_.push_back(0); // each class's TP and FP starts at 0
    class_FPs_.push_back(0);
    class_Totals_.push_back(0);
    // for each class, dynamically declare one top blob
    // this dynamic 'top' declaration is a hack. It is the reason we needed to
    // implement this ctor in the first place. To be consistent with the rest of
    // caffe's way of doing things, we'll leave resizing of top blobs to
    // LayerSetUp()
    // string top_name = param.name()+"."+class_name;
    // const_cast<LayerParameter&>(param).add_top(top_name);
  }
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::custom_test_information() {
  LOG(INFO) << "Per-Class Information for "<< this->layer_param_.name();
  // start the print!
  for(int i = 0; i<num_classes_; i++){
    // we print precision and FN+TP
    float precision = float(class_TPs_[i]) / (class_TPs_[i] + class_FPs_[i]);
    float recall = float(class_TPs_[i]) / class_Totals_[i];
    LOG(INFO)<< class_names_[i] 
        << ": precision=" << precision
        << ", recall=" << recall
        << ", total encountered=" << class_Totals_[i];
    // TODO we should provide a running average sort of statistic for classes
    // that are really rare.
  }

  // after printing, reset tracking information
  for(int i = 0; i<num_classes_; i++){
    class_TPs_[i]=0;
    class_FPs_[i]=0;
    class_Totals_[i]=0;
  }
}
template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
    LOG(INFO)<<"accuracy has ignore_label: " << ignore_label_;//HYQ
  }
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  // HYQ per-class accuracy does not output top
  //vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  //top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  int count = 0;
  // outer_num_ is number of samples per batch
  // inner_num_ is prediction per sample, usually 1. can be W*H also
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      bool match = false;
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          //++accuracy;
          match = true;
          break;
        }
      }
      class_Totals_[label_value]+=1;
      if (match)
        class_TPs_[label_value]+=1;
      else
        class_FPs_[bottom_data_vector[0].second]+=1;
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  //if(count==0)
  //  LOG(INFO)<< "Accuracy cannot be computed with count 0";
  //top[0]->mutable_cpu_data()[0] = count==0? 0 : accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PerClassAccuracyLayer);
REGISTER_LAYER_CLASS(PerClassAccuracy);

}  // namespace caffe
