#!/bin/bash


base_directory=$HOME/Joint_Aspect_Sentiment_Model/Asp_Sent_joint_SemEval_count

# Location of the downloaded data. This is also the place where learned models
# and representations extracted from them will be written. Should have lots of
# space ~30G. EDIT this for your setup.
prefix=$HOME/Joint_Aspect_Sentiment_Model/Asp_Sent_joint_SemEval_count/data

# Amount of gpu memory to be used for buffering data. Adjust this for your GPU.
# For a GPU with 6GB memory, this should be around 4GB.
# If you get 'out of memory' errors, try decreasing this.
gpu_mem=4G

# Amount of main memory to be used for buffering data. Adjust this according to
# your RAM. Having atleast 16G is ideal.
main_mem=20G

# Number of train/valid/test splits for doing classifiation experiments.
numsplits=5

trainer=${base_directory}/trainer.py
extract_rep=${base_directory}/extract_rbm_representation.py
model_output_dir=${prefix}/dbn_models
data_output_dir=${prefix}/dbn_reps
clobber=true

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

# Set up paths, split data into with/without text.
echo Setting up data
python setup_data.py ${prefix} ${model_output_dir} ${data_output_dir} \
  ${gpu_mem} ${main_mem} ${numsplits} || exit 1

# Compute mean and variance of the data.
echo Computing mean / variance
python ${base_directory}/compute_data_stats.py ${prefix}/joint.pbtxt \
    ${prefix}/flickr_stats.npz sent_unlabelled || exit 1
#fi

# SENTIMENT LAYER - 1.
#(
if ${clobber} || [ ! -e ${model_output_dir}/sent_rbm1_LAST ]; then
  echo "Training first layer sentimentRBM."
  python ${trainer} models/sent_rbm1.pbtxt \
    trainers/dbn/train_CD_sent_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/sent_rbm1_LAST \
    trainers/dbn/train_CD_sent_layer1.pbtxt sent_hidden1 \
    ${data_output_dir}/sent_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# SENTIMENT LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/sent_rbm2_LAST ]; then
  echo "Training second layer sentimentRBM."
  python ${trainer} models/sent_rbm2.pbtxt \
    trainers/dbn/train_CD_sent_layer2.pbtxt eval.pbtxt || exit 1
  echo "Extracting RBM representation"
  python ${extract_rep} ${model_output_dir}/sent_rbm2_LAST \
    trainers/dbn/train_CD_sent_layer2.pbtxt sent_hidden2 \
    ${data_output_dir}/sent_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# TEXT LAYER - 1.
if ${clobber} || [ ! -e ${model_output_dir}/text_rbm1_LAST ]; then
  echo "Training first layer text RBM."
  python ${trainer} models/text_rbm1.pbtxt \
    trainers/dbn/train_CD_text_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/text_rbm1_LAST \
    trainers/dbn/train_CD_text_layer1.pbtxt text_hidden1 \
    ${data_output_dir}/text_rbm1_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# TEXT LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/text_rbm2_LAST ]; then
  echo "Training second layer text RBM."
  python ${trainer} models/text_rbm2.pbtxt \
    trainers/dbn/train_CD_text_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/text_rbm2_LAST \
    trainers/dbn/train_CD_text_layer2.pbtxt text_hidden2 \
    ${data_output_dir}/text_rbm2_LAST ${gpu_mem} ${main_mem} || exit 1
fi
#)&
#wait;

# MERGE sentimentAND TEXT DATA PBTXT FOR TRAINING JOINT RBM
if ${clobber} || [ ! -e ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt ]; then
  mkdir -p ${data_output_dir}/joint_rbm_LAST
  python merge_dataset_pb.py \
    ${data_output_dir}/sent_rbm2_LAST/data.pbtxt \
    ${data_output_dir}/text_rbm2_LAST/data.pbtxt \
    ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt || exit 1
fi

# TRAIN JOINT RBM
if ${clobber} || [ ! -e ${model_output_dir}/joint_rbm_LAST ]; then
  echo "Training joint layer RBM."
  python ${trainer} models/joint_rbm.pbtxt \
    trainers/dbn/train_CD_joint_layer.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/joint_rbm_LAST \
    trainers/dbn/train_CD_joint_layer.pbtxt joint_hidden \
    ${data_output_dir}/joint_rbm_LAST ${gpu_mem} ${main_mem} || exit 1
fi

# FINE TUNE THE MODEL.
if ${clobber} || [ ! -e ${data_output_dir}/dbn_all_layers/data.pbtxt ]; then
  echo "Collecting all representations"
  python collect_dbn_reps.py models/multimodal_dbn.pbtxt \
    ${data_output_dir}/dbn_all_layers \
    ${data_output_dir} \
    ${prefix} \
    ${gpu_mem} ${main_mem} || exit 1

# COLLECT ALL LEARNED REPRESENTATION FOR ALL LAYERS IN THE NETWORK
fi
python ${extract_rep} ${model_output_dir}/multimodal_dbn_rbm_LAST trainers/dbn/train_CD_dbn_layer.pbtxt joint_hidden ${data_output_dir}/multimodal_dbn_rbm_LAST ${gpu_mem} ${main_mem}
python ${extract_rep} ${model_output_dir}/multimodal_dbn_rbm_LAST trainers/dbn/train_CD_dbn_layer.pbtxt text_hidden2 ${data_output_dir}/multimodal_dbn_rbm_LAST ${gpu_mem} ${main_mem}
python ${extract_rep} ${model_output_dir}/multimodal_dbn_rbm_LAST trainers/dbn/train_CD_dbn_layer.pbtxt image_hidden2 ${data_output_dir}/multimodal_dbn_rbm_LAST ${gpu_mem} ${main_mem}
python ${extract_rep} ${model_output_dir}/multimodal_dbn_rbm_LAST trainers/dbn/train_CD_dbn_layer.pbtxt image_hidden1 ${data_output_dir}/multimodal_dbn_rbm_LAST ${gpu_mem} ${main_mem}
python ${extract_rep} ${model_output_dir}/multimodal_dbn_rbm_LAST trainers/dbn/train_CD_dbn_layer.pbtxt text_hidden1 ${data_output_dir}/multimodal_dbn_rbm_LAST ${gpu_mem} ${main_mem}

#WE RUN THE NEXT PART ONLY WHEN WE NEED TO DO CLASSIFICATION

fi
echo "SPLIT INTO TRAIN/VALIDATION/TEST SETS FOR CLASSIFICATION."
# SPLIT INTO TRAIN/VALIDATION/TEST SETS FOR CLASSIFICATION.
for i in `seq ${numsplits}`; do
  (
  if ${clobber} || [ ! -e ${data_output_dir}/split_${i}/data.pbtxt ]; then
    python split_reps.py ${data_output_dir}/dbn_all_layers/data.pbtxt \
      ${data_output_dir}/split_${i} ${prefix} ${i} ${gpu_mem} ${main_mem}
  fi
  )&
done
wait;
echo "DO LAYER-WISE CLASSIFICATION"
# DO LAYER-WISE CLASSIFICATION
for i in `seq ${numsplits}`
do
  #(
  for layer in sent_input sent_hidden1 sent_hidden2 joint_hidden text_hidden2 text_hidden1 text_input
  do
    if ${clobber} || [ ! -e ${model_output_dir}/classifiers/split_${i}/${layer}_classifier_BEST ]; then
      echo Split ${i} ${layer}
      python ${trainer} models/classifiers/${layer}_classifier.pbtxt \
        trainers/classifiers/split_${i}.pbtxt eval.pbtxt || exit 1
    fi
  done
  #)&
done
#wait;
# COLLECT RESULTS AND PUT INTO A LATEX TABLE.
if ${clobber} || [ ! -e results.tex ]; then
  python create_results_table.py ${model_output_dir}/classifiers ${numsplits} \
    results.tex || exit 1
fi
