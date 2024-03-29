# Active Learning, Sequence Tagging

Environment is available in `environment.yml`. For completeness, and as a fallback case, the steps used to create it are:
```
conda create -n odinsynth_al
conda activate odinsynth_al
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers
pip install datasets
pip install evaluate
pip install accelerate
pip install sklearn
pip install seaborn
pip install seqeval # Needed for metric computation with evaluate
```
To check the exact versions, see the `environment.yml` file.



### Error while loading the datasets
It is possible to enocunter some errors when loading the dataset.
This is caused by the version of the `datasets` library (see [GitHub Issue](https://github.com/huggingface/datasets/issues/5406)) 
The solution, in this case, is to follow the suggestion in that issue, which is to upgrade the library: 
>For example, versions 2.6.2 and 2.7.1 patch this issue.