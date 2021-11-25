export CUDA_VISIBLE_DEVICES=0

# test J
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold0.yaml -w model_ours/CTRGCN_J_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold1.yaml -w model_ours/CTRGCN_J_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold2.yaml -w model_ours/CTRGCN_J_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold3.yaml -w model_ours/CTRGCN_J_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_J_fold4.yaml -w model_ours/CTRGCN_J_fold4.pdparams

# test JA
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold0.yaml -w model_ours/CTRGCN_JA_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold1.yaml -w model_ours/CTRGCN_JA_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold2.yaml -w model_ours/CTRGCN_JA_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold3.yaml -w model_ours/CTRGCN_JA_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JA_fold4.yaml -w model_ours/CTRGCN_JA_fold4.pdparams

# test B
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold0.yaml -w model_ours/CTRGCN_B_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold1.yaml -w model_ours/CTRGCN_B_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold2.yaml -w model_ours/CTRGCN_B_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold3.yaml -w model_ours/CTRGCN_B_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_B_fold4.yaml -w model_ours/CTRGCN_B_fold4.pdparams

# test BA
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold0.yaml -w model_ours/CTRGCN_BA_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold1.yaml -w model_ours/CTRGCN_BA_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold2.yaml -w model_ours/CTRGCN_BA_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold3.yaml -w model_ours/CTRGCN_BA_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_BA_fold4.yaml -w model_ours/CTRGCN_BA_fold4.pdparams

# test JB
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold0.yaml -w model_ours/CTRGCN_JB_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold1.yaml -w model_ours/CTRGCN_JB_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold2.yaml -w model_ours/CTRGCN_JB_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold3.yaml -w model_ours/CTRGCN_JB_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JB_fold4.yaml -w model_ours/CTRGCN_JB_fold4.pdparams

# test JMj
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold0.yaml -w model_ours/CTRGCN_JMj_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold1.yaml -w model_ours/CTRGCN_JMj_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold2.yaml -w model_ours/CTRGCN_JMj_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold3.yaml -w model_ours/CTRGCN_JMj_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_JMj_fold4.yaml -w model_ours/CTRGCN_JMj_fold4.pdparams

# test Mb
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold0.yaml -w model_ours/CTRGCN_Mb_fold0.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold1.yaml -w model_ours/CTRGCN_Mb_fold1.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold2.yaml -w model_ours/CTRGCN_Mb_fold2.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold3.yaml -w model_ours/CTRGCN_Mb_fold3.pdparams
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_fsd_Mb_fold4.yaml -w model_ours/CTRGCN_Mb_fold4.pdparams

# run ensemble
python3.7 ensemble.py
