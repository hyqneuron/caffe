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
    // now, if normalized, we also give a confusion report
    if(normalize){
      outfile<< (normalize_bylabel? "Recall report " : "Precision report ")
          <<std::endl;
      // one row per class
      for(int i = 0; i<class_names_.size(); i++){

        outfile << format("%20s (%1.3f) %s ") 
                % class_names_[i]
                % (a_to_b_[i][i]/float( normalize_bylabel?
                            class_label_total_[i] : class_pred_total_[i]))
                % (normalize_bylabel? " >> " : " << ");
        // one class per ckeck
        for(int j = 0; j<class_names_.size(); j++){
          float val = 0.0;
          val = normalize_bylabel?
              (a_to_b_[i][j]/(float)class_label_total_[i]) : 
              (a_to_b_[j][i]/(float)class_pred_total_[i])  ;
          // if problem greater than 0.01, we make a report
          if (i!=j && val > 0.01){
              outfile << format(" (%s, %1.3f)") 
                  % class_names_[j]
                  % val;
          }
        }
        outfile << std::endl;
      }
    }
    // we are done
}

// compute 
//   - overall accuracy
//   - mean accuracy (across classes)
// label_total: for this label, how many instances appeared
// x_to_y: confusion matrix
std::pair<float,float> get_accu(
        vector<int> label_total,
        vector<vector<int> > x_to_y) {
  int total_label = 0;
  int total_TP = 0;
  float mean_accu_sum = 0;
  int mean_accu_denom = 0;
  for(int i = 0; i<label_total.size(); i++){
    total_label += label_total[i];
    total_TP += x_to_y[i][i];
    if(label_total[i]!=0){
      mean_accu_denom+=1;
      mean_accu_sum += float(x_to_y[i][i]) / label_total[i];
    }
  }
  return std::make_pair(float(total_TP)/total_label,
                        mean_accu_sum/mean_accu_denom);
}
template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::custom_test_information2() {
  // custom_test_information2
  //     prints overall accuracy and mean accuracy

  // compute true positive rate first
  std::pair<float,float> accus = get_accu(class_label_total_, a_to_b_);
  // print
  string prefix = this->layer_param_.phase()==TRAIN? "TRAIN: " : "TEST: ";
  string name = this->layer_param_.name();
  LOG(INFO) << prefix + "Accuracy for "<< name
              << " = " << accus.first;
  LOG(INFO) << prefix + "Mean Accuracy for "<< name
              << " = " << accus.second;

  // after printing, reset tracking information
  // HACK custom_test_information() and custom_test_information2() must both be
  // called to have correct resetting behaviour
  clear_records();
}
template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::clear_records() {
  // for confusion matrix
  for(int i = 0; i<num_classes_; i++){
    class_label_total_[i]=0;
    class_pred_total_[i]=0;
    a_to_b_[i] = vector<int>(num_classes_);
  }
  // for hierarchical error rate
  if(use_hierarchy_){
    for(int i = 0; i<hier_graded_TP_.size(); i++)
      hier_graded_TP_[i]=0;
    hier_total_ = 0;
    if(use_detailed_hier_accu_){
      vector<int>tmpl;
      tmpl.resize(num_superclass_);
      for(int i = 0; i<num_superclass_;i++){
        suplabel_total_[i]=0;
        suppred_total_[i]=0;
        supa_to_supb_[i]=tmpl;
      }
    }
  }
  // for confusion id
  if(record_confusion_)
    confusion_ids_.clear();
  // for probabilities
  if(record_probabilities_)
    probabilities_.clear();
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::custom_test_information() {
  // custom_test_information 
  //    prints accuracy table (only for TEST phase)
  //    writes confusion file
  if(this->layer_param_.phase()==TEST){
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
  }
  // suffix is the appended to names of files that we output
  string suffix = this->layer_param_.phase()==TRAIN? ".train" : ".test";
  // now, if we need to write confusion matrix to file
  if(this->layer_param_.per_class_accuracy_param().has_confusion_matrix_file()){
    string conf_file = 
        this->layer_param_.per_class_accuracy_param().confusion_matrix_file()
        + suffix;
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
    // print overall accuracy and mean accuracy
    std::pair<float,float> accus = get_accu(class_label_total_, a_to_b_);
    outfile<< std::endl;
    outfile<< "Overall Accuracy: " << accus.first<<std::endl;
    outfile<< "Mean    Accuracy: " << accus.second<<std::endl;
    outfile<< std::endl;
    // print hierarchy-based accuracy
    if(use_hierarchy_){
      outfile <<"Hierarchy-based accuracy"<< std::endl;
      for(int i = 0; i<hier_graded_TP_.size(); i++){
        outfile<< format("  %1.2f  %1.5f")
                     % (float(i)/hier_graded_TP_.size())
                     % (float(hier_graded_TP_[i]) / hier_total_);
        outfile << std::endl;
      }
      // print superclass-based tables
      if(use_detailed_hier_accu_){
        // superclass precision
        outfile << "Precision table for superclass (normalized by TP+FP)" << std::endl;
        printTable(outfile, superclass_names_, supa_to_supb_, 
            suplabel_total_, suppred_total_,
            true, false, true);
        // superclass recall
        outfile << "Recall table for superclass (normalized by TP+FN)" << std::endl;
        printTable(outfile, superclass_names_, supa_to_supb_, 
            suplabel_total_, suppred_total_,
            true, true, false);
        // superclass overall accuracy and mean accuracy
        std::pair<float,float> sup_accus = get_accu(suplabel_total_, supa_to_supb_);
        outfile<< std::endl;
        outfile<< "Superclass Overall Accuracy: " << sup_accus.first<<std::endl;
        outfile<< "Superclass Mean    Accuracy: " << sup_accus.second<<std::endl;
        outfile<< std::endl;
      }
    }
  }
  // if we need to write down the confusion ids
  // format: for each false prediction
  //   ID label_value predicted_label
  if(record_confusion_){
    string cid_file = 
        this->layer_param_.per_class_accuracy_param().confusion_id_file();
    std::ofstream outfile(cid_file.c_str(), std::ofstream::app);
    // one row per entry in confusion_ids_
    for(int i = 0; i<confusion_ids_.size(); i++){
      outfile << format("%8i  %3i %3i")
          % std::get<0>(confusion_ids_[i])
          % std::get<1>(confusion_ids_[i])
          % std::get<2>(confusion_ids_[i])
          << std::endl;
    }
  }
  //
  // if we need to write probability file
  // format: for every sample encountered:
  //   ID label_value prob_1 prob_2 prob_3 ... prob_n
  if(record_probabilities_){
      string prob_file = 
          this->layer_param_.per_class_accuracy_param().probabilities_file();
      std::ofstream outfile(prob_file.c_str(), std::ofstream::app);
      // one row per entry in probabilities_
      for(int i = 0; i<probabilities_.size(); i++){
        outfile << format("%8i %3i ") 
          % std::get<0>(probabilities_[i])
          % std::get<1>(probabilities_[i]);
        // for prob per label
        for(int j = 0; j<std::get<2>(probabilities_[i]).size(); j++){
          outfile << format(" %1.4f") % std::get<2>(probabilities_[i])[j];
        }
        outfile << std::endl;
      }
  }
  // tracking information is reset in custom_test_information2()
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

  use_hierarchy_         = this->layer_param_.per_class_accuracy_param()
                               .use_hierarchy();
  use_detailed_hier_accu_= this->layer_param_.per_class_accuracy_param()
                              .use_detailed_hier_accu();
  record_confusion_      = this->layer_param_.per_class_accuracy_param()
                              .has_confusion_id_file();
  record_probabilities_  = this->layer_param_.per_class_accuracy_param()
                              .has_probabilities_file();
  CHECK(!(use_detailed_hier_accu_ && ! use_hierarchy_))
      << "if using detailed hierarchical accuracy, then must use hierarchy first.";
  if(record_confusion_ || record_probabilities_)
    CHECK_EQ(bottom.size(),3) << "When recording confusion or probability, "
         << "a bottom[2] should provide integral product_id";

  // start reading classifier_info file
  CHECK(this->layer_param_.has_per_class_accuracy_param()) 
      << "PerClassAccuracyLayer requires a per_class_accuracy_param";
  const string& c_file = this->layer_param_.per_class_accuracy_param().classifier_info_file();
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
  // if we're using hierarchy, the thing that comes is the super-class
  // information
  if(use_hierarchy_){
    // first line is the number of superclasses (N)
    // then, followed by 1 line per superclass
    // each superclass's line is:
    // name num_subclasses list_of_subclass_indices
    infile >> num_superclass_;
    // one line per superclass
    for(int i = 0; i<num_superclass_; i++){
      string sup_name;
      int    sup_size;
      int    member_id;
      infile >> sup_name;
      infile >> sup_size;
      CHECK_GT(sup_size, 0) << sup_name << "has no subclass";
      superclass_names_.push_back(sup_name);
      superclass_sizes_.push_back(sup_size);
      vector<int> members;
      for(int j = 0; j<sup_size; j++){
        infile >> member_id;
        members.push_back(member_id);
      }
      superclass_members_.push_back(members);
    }
    // print things out just to show we read the file correctly.
    LOG(INFO)<< this->layer_param_.name() << " has " << num_superclass_
        << " superclasses";
    for(int i = 0; i<num_superclass_; i++){
      string line = superclass_names_[i]+": ";
      for(int j = 0; j<superclass_sizes_[i]; j++)
        line += " "+class_names_[superclass_members_[i][j]];
      LOG(INFO)<< line;
    }
    // set up grades
    for(int i = 0; i<this->layer_param_.per_class_accuracy_param().num_grades();i++)
      hier_graded_TP_.push_back(0);
    hier_total_ = 0;

    // if also using detailed hierarchical accuracy
    if(use_detailed_hier_accu_){
      vector<int> supa_to_supb_template;
      supa_to_supb_template.resize(num_superclass_);
      // set up supa_to_supb_, suplabel_total_, suppred_total_;
      for(int i = 0; i<num_superclass_; i++){
        suplabel_total_.push_back(0);
        suppred_total_.push_back(0);
        supa_to_supb_.push_back(supa_to_supb_template);
      }
    }
  }
  // for record_probabilities_ and record_confusion_, nothing needs to be done
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
  //top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_pid = (record_confusion_||record_probabilities_)?
                             bottom[2]->cpu_data() : NULL;
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
      vector<Dtype> prob_for_hier;
      for (int k = 0; k < num_labels; ++k) {
        Dtype prob = bottom_data[i * dim + k * inner_num_ + j];
        bottom_data_vector.push_back(std::make_pair(prob, k));
        prob_for_hier.push_back(prob);
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
      // compute hierarchical accuracy as an afterthought
      if(use_hierarchy_)
        compute_hierarchical_accuracy(prob_for_hier, predicted_label, label_value);
      int sample_ID = -1;
      if(record_confusion_ || record_probabilities_){
        sample_ID = int(bottom_pid[i]);
      }
      if(record_confusion_ && label_value != predicted_label){
        confusion_ids_.push_back(
                std::make_tuple(sample_ID, label_value, predicted_label));
      }
      if(record_probabilities_){
        probabilities_.push_back(
                std::make_tuple(sample_ID, label_value, vector<Dtype>(
                        bottom_data+i*dim, bottom_data+(i+1)*dim)));
      }
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  // if(count==0)
  //  LOG(INFO)<< "Accuracy cannot be computed with count 0 for: " 
  //           << this->layer_param_.name();
  // top[0]->mutable_cpu_data()[0] = count==0? 0 : accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::compute_hierarchical_accuracy(
        vector<Dtype> probs, int predicted_label, int label_value
        ){
  // data needed: prob, superclass_xxx, max_index
  //
  // Steps:
  // - Compute prob_max, prob_sum, prob_mean of each superclass
  // - Find index of superclasses with maximal prob_max, prob_sum and prob_mean
  // - determine if the superclasses that correspond to max_sum and max_mean
  //   contain label_value
  if(num_superclass_<=0) return;

  vector<Dtype> sup_prob_max;
  vector<Dtype> sup_prob_sum;
  vector<Dtype> sup_prob_mean;
  int max_max_index  = 0;
  int max_sum_index  = 0;
  int max_mean_index = 0;
  Dtype max_max  = 0;
  Dtype max_sum  = 0;
  Dtype max_mean = 0;
  // find maximal max, sum, and mean
  for(int i = 0; i<num_superclass_; i++){
    Dtype max = 0;
    Dtype sum = 0;
    Dtype mean= 0;
    for(int j = 0; j<superclass_sizes_[i]; j++){
      Dtype p = probs[superclass_members_[i][j]];
      sum += p;
      if (p>max) max=p;
    }
    mean = sum/superclass_sizes_[i];
    if(max>max_max){
      max_max = max;
      max_max_index = i;
    }
    if(sum>max_sum){
      max_sum = sum;
      max_sum_index = i;
    }
    if(mean>max_mean){
      max_mean = mean;
      max_mean_index = i;
    }
    sup_prob_max.push_back(max);
    sup_prob_sum.push_back(sum);
    sup_prob_mean.push_back(mean);
  }
  // ====== THIS SECTION FOR simple hierarchical accuracy ======
  // find if they contain it
  vector<int>::iterator found;
  bool max_sum_haslabel, max_mean_haslabel;
  // for max_sum
  vector<int> &target = superclass_members_[max_sum_index];
  found=std::find( target.begin(), target.end(), label_value);
  max_sum_haslabel = found != target.end();
  // for max_mean
  target = superclass_members_[max_mean_index];
  found=std::find( target.begin(), target.end(), label_value);
  max_mean_haslabel = found != target.end();

  hier_total_ += 1;
  Dtype predicted_prob = probs[predicted_label];
  for(int i = 0; i<hier_graded_TP_.size(); i++){
    // We make i=0 a special case, where we consider the prob_sum, not prob_mean
    //
    // when i=0, we consider
    //      max_sum  > predicted_prob
    // when i!=0, we consider the relationship of
    //      max_mean > k * predicted_prob
    //      where k = i/hier_grade_TP_.size() (if got 10 grades,then 0.1 to 0.9)
    //
    // if the inequality is true, we predict the corresponding superclass
    // otherwise we predict the original subclass
    Dtype max_val = i==0? max_sum : max_mean;
    Dtype val_cap = i==0? 
                predicted_prob : 
                predicted_prob * i / float(hier_graded_TP_.size());
    bool max_haslabel = i==0? max_sum_haslabel : max_mean_haslabel;
    // predict superclass
    if(max_val>=val_cap){
      if(max_haslabel)
        hier_graded_TP_[i]++;
    }
    // predict subclass
    else if(predicted_label==label_value)
      hier_graded_TP_[i]++;
  }
  // ====== Detailed hierarchical accuracy ======
  if(use_detailed_hier_accu_){
    // - find true superclass
    // - find predicted superclass
    // - record values into suplabel suppred and supa_to_supb_

    // find true superclass
    //   since we don't have something that maps subclass to superclass, we'll
    //   have to do lookup... which is quite stupid, but easy to code
    int label_sup = -1;
    for(int i = 0; i<num_superclass_; i++){
      target = superclass_members_[i];
      found=std::find( target.begin(), target.end(), label_value);
      if( found != target.end()){
        label_sup = i;
        break;
        // we assume the superclasses are mutually exclusive, so once it is
        // found, we break
      }
    }
    CHECK_NE(label_sup, -1) <<
        "Encountered a label that does not belong to any superclass when using detailed hierarchical accuracy."
        <<std::endl
        <<"label="<<label_value
        << std::endl
        <<"classifier="<<classifier_name_;

    // find predicted superclass
    //   we take a very simple policy: he who has max_max is the predicted
    //   superclass
    int pred_sup = max_max_index;

    // record values
    supa_to_supb_[label_sup][pred_sup]++;
    suplabel_total_[label_sup]++;
    suppred_total_[pred_sup]++;

  }
}

INSTANTIATE_CLASS(PerClassAccuracyLayer);
REGISTER_LAYER_CLASS(PerClassAccuracy);

}  // namespace caffe
