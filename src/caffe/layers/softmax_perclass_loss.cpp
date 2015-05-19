#include <algorithm>
#include <cfloat>
#include <vector>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <math.h>
#include "boost/format.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithPerClassLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // check ignored label and normalize_ if necessary
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
    LOG(INFO)<<"loss has ignore_label: " << ignore_label_;//HYQ
  }
  normalize_ = this->layer_param_.loss_param().normalize();

  class_specific_lr_ = softmax_param.loss_param().class_specific_lr();
  // start reading file
  // adapted from per_class_accuracy_layer.cpp
  CHECK(softmax_param.loss_param().has_classifier_info_file()) <<
      "SoftmaxWithPerClassLossLayer requires 'classifier_info_file' to be specified in softmax_param";
  const string& c_file = softmax_param.loss_param().classifier_info_file();
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
  for (int i=0;i<num_classes_;i++){
    infile >> class_label;
    infile >> class_name;
    infile >> class_prior;
    infile >> class_lrmult;
    class_labels_.push_back (class_label);
    class_names_.push_back  (class_name);
    class_priors_.push_back (class_prior);
    class_lrmults_.push_back(class_lrmult);
  }
}

template <typename Dtype>
void SoftmaxWithPerClassLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  // outer_num_ = number of samples
  // inner_num_ = W*H in (N,C,W,H), and is usually 1
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  // set up lr_mult
  vector<int> lr_mult_shape;
  lr_mult_shape.push_back(outer_num_);
  lr_mult_.Reshape(lr_mult_shape);
}

template <typename Dtype>
void SoftmaxWithPerClassLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  // outer_num_ is number of samples
  // for simplicity, assume dim = number of classes
  //                 assume inner_num_ = 1
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = count==0? 0 : loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = outer_num_==0? 0 : loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithPerClassLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    // HYQ this is where we cut in. Instead of scaling it by a per-batch
    // constant, we should scale each bottom_diff[i*inner_num_+j] by
    // class_mult_[label_value]
    //
    // for CPU, we just launch outer_num_ caffe_scal ops
    // We assume bottom_diff is structured as [outer_num_][dim]
    const Dtype denominator = normalize_? count : outer_num_;
    for (int i = 0; i < outer_num_; ++i){
      caffe_scal(dim,
                 denominator * class_lrmults_[label[i]],
                 &bottom_diff[i*dim]);
    }
    // THE above should just work.
    // HACK but it does not handle the W*H non-trivial case
    // and I really don't understand the W*H case yet

    /*
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), 
                 count==0? 0 : loss_weight / count, 
                 bottom_diff);
    } else {
      caffe_scal(prob_.count(), 
                 outer_num_==0? 0: loss_weight / outer_num_, 
                 bottom_diff);
    }
    */
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithPerClassLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithPerClassLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithPerClassLoss);

}  // namespace caffe
