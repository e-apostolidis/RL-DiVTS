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
    /top3_thumbnail_ids       the indices of the three most selected thumbnails by the human annotators (can be more than three if they exist more than three key-frames with the same ranking according to the number of selections made by the human annotators.
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
`--selected_thumbs` | Number of selected thumbnails. | 6 | int > 0
`--n_epochs` | Number of training epochs. | 150 | int > 0
`--batch_size` | Size of the training batch, 40 for 'OVP' and 32 for 'Youtube'. | 40 | 0 < int ≤ len(Dataset)
`--seed` | Chosen number for generating reproducible random numbers. | None | None, int
`--exp` | Experiment serial number. | 1000 | int
`--clip` | Gradient norm clipping parameter. | 5.0 | float 
`--lr` | Value of the adopted learning rate. | 1e-4 | float
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 9

## Training
<div align="justify">

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the [data/splits](/data/splits) directory. This file contains the 5 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```bash
python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name' --exp ID
```
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, `dataset_name` refers to the name of the used dataset, and `ID` refers to index of the folder that will be used for storing the analysis results (default naming format: expID). For example, to run an experiment using the _first_ data split of the _OVP_ dataset, a batch of _40_ videos (full-batch), for _150_ training epochs, and store the analysis results in the _exp1_ folder, execute the following command
```bash
python model/main.py --split_index 0 --n_epochs 150 --batch_size 40 --video_type 'OVP' --exp 1
```

Alternatively, to train the model for all 5 splits and all 5 different seeds, use the [`run_ovp_splits.sh`](model/run_ovp_splits.sh) and/or [`run_youtube_splits.sh`](model/run_youtube_splits.sh) script and do the following:
```shell-script
chmod +x model/run_ovp_splits.sh        # Makes the script executable.
chmod +x model/run_youtube_splits.sh    # Makes the script executable.
sh /model/run_ovp_splits.sh             # Runs the script. 
sh /model/run_youtube_splits.sh         # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](evaluation) scripts to assess the overall performance of the model.

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a Terminal and executing the following command: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- and then opening a browser and pasting the returned URL after the execution of the above command in the Terminal 
</div>

## Evaluation and Model Selection 
<div align="justify">

Given a test video, the top-3 selected key-frames among all annotators for this video are considered as the ground-truth thumbnails. As a side note, through this procedure some videos are associated with more than 3 ground-truth thumbnails, due to the existence of more than 3 key-frames with the same ranking according to the number of selections made by the human annotators. Nevertheless, in our evaluations we use the three thumbnails that come first according to the MSD (Most Significant Digit) Radix Sort of Python (which, e.g., sorts frame `#20` before frame `#3` based on the most significant digit).

In terms of evaluation, we applied the "top-3 matching" approach that measures the overlap between the top-3 machine- and human-selected thumbnails per video. We expressed this overlap as a scalar ranging in $[0,1]$ and computed the average score over all videos of the test set.

The utilized model selection criterion relies on the maximization of the received reward and enables the selection of a well-trained model by indicating the training epoch. In [evaluation](evaluation) we provide [evaluate_all_exp.sh](evaluation/evaluate_all_exp.sh) to evaluate the trained models of the architecture and automatically select a well-trained one, for each conducted experiment. To run this script, define:
 - the `h5_file_path` in [`compute_score.py`](evaluation/compute_score.py#L17),
 - the `base_path` in [`evaluate_all_exp.sh`](evaluation/evaluate_all_exp.sh#L8),
 - the `init_id` in [`evaluate_all_exp.sh`](evaluation/evaluate_all_exp.sh#L9),
 - and the 'dataset_name' in [`evaluate_all_exp.sh`](evaluation/evaluate_all_exp.sh#L10)

and run
```bash
sh evaluation/evaluate_all_exp.sh '$exp_id' '$dataset_name' 
```
where, `$exp_id` is the ID of the first (out of five in total) evaluated experiment, and `$dataset_name` refers to the dataset being used.

## Citation
<div align="justify">

If you find our work or code, useful in your work, please cite the following publication:

E. Apostolidis, G. Balaouras, V. Mezaris, I. Patras, "<b>Selecting a Diverse Set of Aesthetically-Pleasing and Representative Video Thumbnails Using Reinforcement Learning</b>", submitted for publication at the 2023 IEEE Int. Conf. on Image Processing (ICIP 2023).
</div>

## License
<div align="justify">

Copyright © 2023, Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-832921 MIRROR, and by EPSRC under grant No. EP/R026424/1. </div>
