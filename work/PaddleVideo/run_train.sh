export CUDA_VISIBLE_DEVICES=0

# train J
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold4.yaml --validate

# train JA
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold4.yaml --validate

# train B
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold4.yaml --validate

# train BA
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold4.yaml --validate

# train JB
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold4.yaml --validate

# train JMj
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold4.yaml --validate

# train Mb
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold0.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold1.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold2.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold3.yaml --validate
python3.7 main.py -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold4.yaml --validate
