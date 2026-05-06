# CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels

This is the source code for the paper: [CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels][https://icml.cc/virtual/2026/poster/65515] (ICML 2026).

## Requirements

* Python 3.8
* PyTorch 2.0
* Torchvision 0.15
* Tensorboard

- Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n lift python=3.8 -y
conda activate lift
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Hardware

Most experiments can be reproduced using a single GPU with 24GB of memory.

## Quick Start on the CIFAR-100-LTN dataset

```bash
# run CARE on CIFAR-100-LTN (with imbalance ratio=100, noise ratio=50)
python main.py -d cifar100_ir100_nr50 -m clip_vit_b16 adaptformer True
```

By running the above command, you can automatically download the CIFAR-100 dataset and run the method (CARE).

### Detailed Usage

To train and test the proposed method on more settings, run

```bash
python main.py -d [data] -m [model] [options]
```

The `[data]` can be the name of a .yaml file in [configs/data](configs/data), including `webvision-50`, `mini-imagenet`, `food101n`, `cifar100_ir100_50`, `cifar100_ir10_nr50`, etc.

The `[model]` can be the name of a .yaml file in [configs/model](configs/model), including `clip_rn50`, `clip_vit_b16`, `in21k_vit_b16`, etc.

Note that using only `-d` and `-m` options denotes only fine-tuning the classifier. Please use additional `[options]` for more settings. 

- To apply lightweight fine-tuning methods, add options like `lora True`, `adaptformer True`, etc.

- To apply test-time ensembling, add `tte True`.

Moreover, `[options]` can facilitate modifying the configure options in [utils/config.py](utils/config.py). Following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.

- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.

- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.

- To apply gradient accumulation, add `micro_batch_size XX`. This can further reduce GPU memory costs. Note that `XX` should be a divisor of `batch_size`.

- To test an existing model, add `test_only True`. This option will test the model trained by your configure file. To test another model, add an additional option like `model_dir output/AnotherExpDir`.

- To test an existing model on the training set, add `test_train True`.

## Acknowledgment

We thank the authors for the following repositories for code reference:
[[LIFT]](https://github.com/shijxcs/LIFT). 

## Citation

If you find this repo useful for your work, please cite as:

```bibtex
@inproceedings{li2026care,
  title={CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels},
  author={Meng-Ke Li and Hai-Quan Ling and Li-Hao Chen and Yang Lu and Yi-Qun Zhang and Hui Huang},
  booktitle={Proceedings of the 43st International Conference on Machine Learning},
  year={2026}
}
```
