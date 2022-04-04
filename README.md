# RL-DiVTS: Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning

## PyTorch Implementation of RL-DiVTS  
<div align="justify">

- From **"RL-DiVTS: Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning"**.
- Written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras.

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.3` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11600K` CPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.8(.8) | 1.7.1 | 11.0 | 8005 | 2.4.0 | 2.4.1 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the OVP and YouΤube datasets are available within the [data](data) folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre><code>/key
    /features                 2D-array with shape (n_steps, feature-dimension), feature vectors representing the content of the video frames; extracted from the pool5 layer of a GoogleNet trained on the ImageNet dataset
    /aesthetic_scores_mean    1D-array with shape (n_steps), scores representing the aesthetic quality of the video frames; computed as the softmax of the values in the final layer of a model of a <a href="https://github.com/bmezaris/fully_convolutional_networks" target="_blank">Fully Convolutional Network</a> trained on the AVA dataset
    /n_frames                 number of frames in original video
    /ssim_matrix              2D-array with shape (top-5 selected thumbs, n_frames), the structural similarity scores between each of the five most selected thumbnails by the human annotators (in order to support evaluation using 'Precision at 5') and the entire frame sequence; computed using the <a href="https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity" target="_blank">structural_similarity function</a> of Python
    /top1_thumbnail_ids       the index of the most selected thumbnail by the human annotators (can be more than one if they exist more than one key-frames with the same ranking according to the number of selections made by the human annotators)
    /top3_thumbnail_ids       the indices of the three most selected thumbnails by the human annotators (can be more than three if they exist more than three key-frames with the same ranking according to the number of selections made by the human annotators)
</code></pre>
Original videos and annotations for each dataset are also available here: 
- <a href="https://sites.google.com/site/vsummsite/download" target="_blank"><img align="center" src="https://img.shields.io/badge/Datasets-OVP,%20YouTube-green"/></a>
</div>
 
 ## Configurations
<div align="justify">

Setup for the training process:
 - In [`data_loader.py`](model/data_loader.py#L19:L21), specify the path to the h5 file of the used dataset, and the path to the JSON file containing data about the utilized data splits.
 - In [`configs.py`](model/configs.py#L7), define the directory where the analysis results will be saved to. </div>
   
Arguments in [`configs.py`](model/configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'OVP' | 'OVP', 'Youtube'
`--input_size` | Size of the input feature vectors. | 1024 | int > 0
`--hidden_size` |Number of features in the LSTM hidden state. | 512 | int > 0
`--num_layers` | Number of LSTM recurrent layers. | 2 | int > 0
`--n_episodes` | Number of training episodes per epoch. | 10 | int > 0
`--selected_thumbs` | Number of selected thumbnails. | 5 | int > 0
`--n_epochs` | Number of training epochs. | 200 | int > 0
`--batch_size` | Size of the training batch, 40 for 'OVP' and 32 for 'Youtube'. | 40 | 0 < int ≤ len(Dataset)
`--seed` | Chosen number for generating reproducible random numbers. | None | None, int
`--exp` | Experiment serial number. | 1000 | int
`--clip` | Gradient norm clipping parameter. | 5.0 | float 
`--lr` | Value of the adopted learning rate. | 1e-4 | float
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 9

## Training
<div align="justify">

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the [data/splits](/data/splits) directory. This file contains the 10 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```bash
python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name' --exp ID
```
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, `dataset_name` refers to the name of the used dataset, and `ID` refers to the experimental serial number.

Alternatively, to train the model for all 10 splits, use the [`run_ovp_splits.sh`](model/run_ovp_splits.sh) and/or [`run_youtube_splits.sh`](model/run_youtube_splits.sh) script and do the following:
```shell-script
chmod +x model/run_ovp_splits.sh        # Makes the script executable.
chmod +x model/run_youtube_splits.sh    # Makes the script executable.
./model/run_ovp_splits.sh               # Runs the script. 
./model/run_youtube_splits.sh           # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](evaluation) scripts to assess the overall performance of the model.

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- opening a browser and pasting the returned URL from cmd. </div>

## Model Selection and Evaluation 
<div align="justify">

The utilized model selection criterion relies on the post-processing of the calculated losses over the training epochs and enables the selection of a well-trained model by indicating the training epoch, with the highest overall reward. To evaluate the trained models of the architecture and automatically select a well-trained one, define:
 - the [`h5_file_path`](evaluation/compute_P%40k.py#L12) in [`compute_P@k.py`](evaluation/compute_P%40k.py),
 - the [`h5_file_path`](evaluation/compute_P%40k_on1thumb.py#L12) in [`compute_P@k_on1thumb.py`](evaluation/compute_P%40k_on1thumb.py),
 - the [`base_path`](evaluation/evaluate_exp.sh#L9) in [`evaluate_exp.sh`](evaluation/evaluate_exp.sh),

and run [`evaluate_exp.sh`](evaluation/evaluate_exp.sh) via
```bash
sh evaluation/evaluate_exp.sh '$exp_id' '$dataset_name'
```
where, `$exp_id` is the ID of the current evaluated experiment, and `$dataset_name` refers to the dataset being used.

For further details about the adopted structure of directories in our implementation, please check line [#13](evaluation/evaluate_exp.sh#L13) and line [#17](evaluation/evaluate_exp.sh#L17) of [`evaluate_exp.sh`](evaluation/evaluate_exp.sh). </div>
