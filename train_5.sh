python3 train.py --config ./config/cifar100/118-stage-1_2_1_3_2-ratio-0.5_0.5_0.75_0.5_0.25-op-vgg_vgg_vgg_repvgg_repvgg-pool-False_False_False_False_False_True-pool_type-None_None_None_None_None_avgpool-cifar100.yaml
python3 train.py --config ./config/cifar100/119-stage-2_2_1_1_1-ratio-0.625_0.625_0.625_1.0_0.125-op-vgg_vgg_repvgg_repvgg_repvgg-pool-False_False_False_True_True_True-pool_type-None_None_None_maxpool_avgpool_maxpool-cifar100.yaml