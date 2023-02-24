# Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning

## PyTorch Implementation of RL-DiVTS  
<div align="justify">

- From **"Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning"**.
- Written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras.
- This software can be used for training a deep learning architecture for video thumbnail selection, which quantifies the representativeness and the aesthetic quality of the selected thumbnails using deterministic reward functions, and integrates a frame picking mechanism that takes frames’ diversity into account. After being trained on a collection of videos, `RL-DiVTS`'s Thumbnail Selector  is capable of selecting a diverse set of representative and aesthetically-pleasing video thumbnails for unseen videos, according to a user-specified value about the number of required thumbnails.
- The PyTorch Implementation of the `ARL-VTS` video thumbnail selection method, that is also evaluated in the paper, is available at [`ARL-VTS`](https://github.com/e-apostolidis/Video-Thumbnail-Selector).
</div>

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.5` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11600K` CPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.8(.8) | 1.7.1 | 11.4 | 8005 | 2.4.0 | 2.4.1 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the OVP and YouΤube datasets are available within the [data](data) folder. These files have the following structure:
<pre><code>/key
    /features                 2D-array with shape (n_steps, feature-dimension), feature vectors representing the content of the video frames; extracted from the pool5 layer of a GoogleNet trained on the ImageNet dataset
    /aesthetic_scores_mean    1D-array with shape (n_steps), scores representing the aesthetic quality of the video frames; computed as the softmax of the values in the final layer of a model of a <a href="https://github.com/bmezaris/fully_convolutional_networks" target="_blank">Fully Convolutional Network</a> trained on the AVA dataset
    /n_frames                 number of frames in original video
    /ssim_matrix              2D-array with shape (M, n_frames), the structural similarity scores between each of the M most selected thumbnails by the human annotators and the entire frame sequence (to support evaluation using 'Precision at 5' the number of selected thumbnails by the human annotators was set equal to five; however, M can be more than five if they exist more than five key-frames with the same ranking according to the number of selections made by the human annotators); the structural similarity scores were computed using the <a href="https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity" target="_blank">structural_similarity function</a> of Python
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
 - In [`configs.py`](model/configs.py#L7), define the directory where the analysis results will be saved to.
</div>
 
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
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, `dataset_name` refers to the name of the used dataset, and `ID` refers to index of the folder that will be used for storing the analysis results (default naming format: expID). For example, to run an experiment using the _first_ data split of the _OVP_ dataset, a batch of _40_ videos (full-batch), for _100_ training epochs, and store the analysis results in the _exp10_ folder, execute the following command
```bash
python model/main.py --split_index 0 --n_epochs 100 --batch_size 40 --video_type 'OVP' --exp 10
```

Alternatively, to train the model for all 10 splits, use the [`run_ovp_splits.sh`](model/run_ovp_splits.sh) and/or [`run_youtube_splits.sh`](model/run_youtube_splits.sh) script and do the following:
```shell-script
chmod +x model/run_ovp_splits.sh        # Makes the script executable.
chmod +x model/run_youtube_splits.sh    # Makes the script executable.
./model/run_ovp_splits.sh               # Runs the script. 
./model/run_youtube_splits.sh           # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](evaluation) scripts to assess the overall performance of the model. Last but not least, the above mentioned [`run_ovp_splits.sh`](model/run_ovp_splits.sh) and/or [`run_youtube_splits.sh`](model/run_youtube_splits.sh) scripts allow the execution of multiple runs using the set of 10 splits, in order to facilitate the implementation of the adopted evaluation strategy that was based on 10 different runs.

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a Terminal and executing the following command: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- and then opening a browser and pasting the returned URL after the execution of the above command in the Terminal 
</div>

## Evaluation and Model Selection 
<div align="justify">

Given a test video, the top-3 selected key-frames among all annotators for this video are considered as the ground-truth thumbnails. As a side note, through this procedure some videos are associated with more than 3 ground-truth thumbnails, due to the existence of more than 3 key-frames with the same ranking according to the number of selections made by the human annotators. 

In terms of evaluation, we considered two different scenarios. In the first scenario, the goal of video thumbnail selection is defined as having among the machine-selected thumbnails at least one that matches one of the user-selected (ground-truth) ones. `Acc@k/GT=λ` declares a hit if at least one of the `k` machine-selected thumbnails matches one (or more) of the `λ` ground-truth ones (i.e., their computed SSIM score is above a predefined threshold, that equals to 0.7). The lower the values of `k`, `λ` are, the more challenging this evaluation scenario becomes. In a second, even more challenging evaluation scenario, assuming `k = λ > 1`, the goal is to maximize the number of machine-selected thumbnails that match one (or more) of the ground-truth ones. This is equal to estimating `Acc@k/GT=λ` separately for each of the `k` machine-selected thumbnails, and averaging these results over `k`.<sup name="a1">[1](#f1)</sup> We denote this as `AVGAcc@k/GT=λ`.

The utilized model selection criterion relies on the the maximization of the received reward and enables the selection of a well-trained model by indicating the training epoch. In [evaluation](evaluation) we provide [evaluate_exp.sh](evaluation/evaluate_exp.sh) to evaluate the trained models of the architecture and automatically select a well-trained one, for a given experiment. Furthermore, [evaluate_all_exp.sh](evaluation/evaluate_all_exp.sh) can be used to enable the reproducibillity of our findings over 10 experimental runs. Finally, to run the abovementioned scripts, define:
 - the `h5_file_path` in [`compute_score_single.py`](evaluation/compute_score_single.py#L21) and/or [`compute_score.py`](evaluation/compute_score.py#L21),
 - the `base_path` in [`evaluate_exp.sh`](evaluation/evaluate_exp.sh#L9) and/or [`evaluate_all_exp.sh`](evaluation/evaluate_all_exp.sh#L9),

and run
```bash
    sh evaluation/evaluate_exp.sh '$exp_id' '$dataset_name' 'precision'  # for Acc@k/GT=λ
    sh evaluation/evaluate_exp.sh '$exp_id' '$dataset_name' 'matching'   # for AVGAcc@k/GT=λ
```
where, `$exp_id` is the ID of the current evaluated experiment, and `$dataset_name` refers to the dataset being used. The [`evaluate_all_exp.sh`](evaluation/evaluate_all_exp.sh) follows the same logic in arguments.

<sup name="f1">1</sup> In this scenario, for the videos where there are more than one key-frames with the same ranking in one of the top-`λ` positions according to the number of selections made by the human annotators, we resolve draws (that would lead to more than `λ` ground-truth thumbnails) by keeping the frame(s) that come first according to the MSD (Most Significant Digit) Radix Sort of Python (which, e.g., sorts frame `#20` before frame `#3` based on the most significant digit). [↩](#a1) 
</div>

## Citation
<div align="justify">

If you find our work or code, useful in your work, please cite the following publication:

E. Apostolidis, G. Balaouras, V. Mezaris, I. Patras, "<b>Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning</b>", submitted for publication at the Int. Journal of Multimedia Information Retrieval.
</div>

## License
<div align="justify">

Copyright © 2022, Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-832921 MIRROR, and by EPSRC under grant No. EP/R026424/1. </div>
