#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <math.h>
#include "boost/format.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using boost::format;

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
  for (int i=0;i<num_classes_;i++){
    infile >> class_label;
    infile >> class_name;
    infile >> class_prior;
    infile >> class_lrmult;
    class_labels_.push_back (class_label);
    class_names_.push_back  (class_name);
    class_priors_.push_back (class_prior);
    class_lrmults_.push_back(class_lrmult);

    // confusion statistics
    class_label_total_.push_back(0);
    class_pred_total_.push_back(0);
    a_to_b_.push_back(vector<int>(num_classes_));
  }
}

void printTable(std::ostream& outfile,
                const vector<string>& class_names_,
                const vector<vector<int> >& a_to_b_,
                const vector<int>& class_label_total_,
                const vector<int>& class_pred_total_,
                bool normalize, 
                bool normalize_bylabel,
                bool normalize_bypred)
{
    // if normalize, must either bylabel or bypred
    CHECK(!normalize || normalize_bylabel || normalize_bypred );
    // we print something like below:
    //           0  1  2  3  4  5  6  7
    // 0 name   nn nn nn
    // 1 name
    // 2 name
    outfile << format("%20s") % " "; // print 20 spaces
    // print top-line indices
    for(int i = 0; i<class_names_.size(); i++)
      outfile << format(" %3i") % i;
    outfile << std::endl;
    // one class per line
    for(int i = 0; i<class_names_.size(); i++){
      outfile << format("%2i %17s") % i % class_names_[i];
      // one number per class on this line
      for(int j = 0; j<class_names_.size(); j++){
        // depending on normalization, we print different number
        if(!normalize)
          outfile << format(" %3i") % a_to_b_[i][j];
        else{
          float val = 0.0;
          val = normalize_bylabel?
            (a_to_b_[i][j]/(float)class_label_total_[i]) : 
            (a_to_b_[i][j]/(float)class_pred_total_[j])  ;
          if(isnan(val)) outfile << " NaN";
          else           outfile << format(" %3i") % int(1000*val);
        }
      }
      outfile << std::endl;
    }
    // we are done
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::custom_test_information() {
  LOG(INFO) << "#############";
  LOG(INFO) << "Per-Class Information for "<< this->layer_param_.name();
  // start the print!
  for(int i = 0; i<num_classes_; i++){
    // we print precision and FN+TP
    float precision = float(a_to_b_[i][i]) / class_pred_total_[i];
    float recall = float(a_to_b_[i][i])    / class_label_total_[i];
    // name TP FP label precision recall
    LOG(INFO)<< format("%20s %10i %10i %10i %10f %10f")
        % class_names_[i]
        % a_to_b_[i][i]
        % (class_pred_total_[i] - a_to_b_[i][i])
        % class_label_total_[i]
        % precision
        % recall;
  }
  // now, if we need to write confusion matrix to file
  if(this->layer_param_.per_class_accuracy_param().has_confusion_matrix_file()){
    string conf_file = 
        this->layer_param_.per_class_accuracy_param().confusion_matrix_file();
    std::ofstream outfile(conf_file.c_str());
    outfile<< format("###################%=20s####################")
            % this->layer_param_.name();
    outfile << std::endl;
    // print raw numbers
    outfile << "Raw number table" << std::endl;
    printTable(outfile, class_names_, a_to_b_, 
        class_label_total_, class_pred_total_,
        false, false, false);
    // print precision (normalized by pred)
    outfile << "precision table (normalized by TP+FP)" << std::endl;
    printTable(outfile, class_names_, a_to_b_, 
        class_label_total_, class_pred_total_,
        true, false, true);
    // print recall (normalized by label)
    outfile << "recall table (normalized by TP+FN)" << std::endl;
    printTable(outfile, class_names_, a_to_b_, 
        class_label_total_, class_pred_total_,
        true, true, false);
  }

  // after printing, reset tracking information
  for(int i = 0; i<num_classes_; i++){
    class_label_total_[i]=0;
    class_pred_total_[i]=0;
    a_to_b_[i] = vector<int>(num_classes_);
  }
}
template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.per_class_accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.per_class_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.per_class_accuracy_param().ignore_label();
    LOG(INFO)<<"accuracy has ignore_label: " << ignore_label_;//HYQ
  }
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.per_class_accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  int count = 0;
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
      int predicted_label = bottom_data_vector[0].second;
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          break;
        }
      }
      CHECK_LT(label_value, num_classes_) << "bad label";
      CHECK_LT(predicted_label, num_classes_) << 
          "number of classes:" << num_classes_ <<
          "predicted label:" << predicted_label;
      // record the stats
      class_label_total_[label_value]+=1;      // number of this label encountered
      class_pred_total_[predicted_label]+=1;   // number of this pred encountered
      a_to_b_[label_value][predicted_label]+=1;// label_to_pred confusion matrix
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  if(count==0)
    LOG(INFO)<< "Accuracy cannot be computed with count 0 for" 
             << this->layer_param_.name();
  top[0]->mutable_cpu_data()[0] = count==0? 0 : accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PerClassAccuracyLayer);
REGISTER_LAYER_CLASS(PerClassAccuracy);

}  // namespace caffe
