# Original
https://github.com/garyzhao/SemGCN

# setup
## environment
- windows 10
- anaconda python 3.7.9
- cuda 10.2

## data setup
```
python prepare_data_h36m.py --from-archive h36m.zip
python prepare_data_2d_h36m_sh.py -pt h36m.zip
python prepare_data_2d_h36m_sh.py -ft stacked_hourglass_fined_tuned_240.tar.gz
```
