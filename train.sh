# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train.py -c conf/cifar100/4_384_300E_t4.yml -data-dir /dataset/CIFAR100/ --model sdt --spike-mode lif

# CUDA_VISIBLE_DEVICES=4 python train.py -c conf/cifar100/2_512_300E_t4_Const_E2.yml -data-dir /dataset/CIFAR100/ --model sdt --spike-mode lif

# CUDA_VISIBLE_DEVICES=5 python train.py -c conf/cifar100/2_512_300E_t4_Const_E4.yml -data-dir /dataset/CIFAR100/ --model sdt --spike-mode lif

# CUDA_VISIBLE_DEVICES=6 python train.py -c conf/cifar100/2_512_300E_t4_Const_E8.yml -data-dir /dataset/CIFAR100/ --model sdt --spike-mode lif


#E2: 78.65 (epoch 208)
#E4: 78.68 (epoch 205)
#E8: 79.02 (epoch 195)


python finetune_routerKD.py -c conf/cifar100/4_384_300E_t4_KDft.yml \
  --finetune /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar 


CUDA_VISIBLE_DEVICES=2 python train_routerKD.py -c conf/cifar100/4_384_300E_t4_KDscratch_bal.yml \
  --experiment "cifar100_scratch_routerKD_bal