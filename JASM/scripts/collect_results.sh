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


# COLLECT RESULTS AND PUT INTO A LATEX TABLE.
echo "COLLECT RESULTS AND PUT INTO A LATEX TABLE."
if ${clobber} || [ ! -e results.tex ]; then
  python create_results_table.py ${model_output_dir}/classifiers ${numsplits} \
    results.tex || exit 1
fi
