python3 train.py --config ./config/cifar100/215-stage-1_4_5_4_2-ratio-0.25_0.25_0.75_0.625_0.25-op-repvgg_vgg_repvgg_repvgg_vgg-pool-False_False_True_False_False_False-pool_type-None_None_maxpool_None_None_None-cifar100.yaml
python3 train.py --config ./config/cifar100/216-stage-1_4_1_6_1-ratio-0.5_0.5_0.75_1.0_0.75-op-repvgg_repvgg_repvgg_vgg_vgg-pool-False_False_False_False_False_True-pool_type-None_None_None_None_None_avgpool-cifar100.yaml
python3 train.py --config ./config/cifar100/217-stage-2_3_2_7_4-ratio-1.0_1.0_1.0_0.875_0.625-op-repvgg_repvgg_repvgg_repvgg_repvgg-pool-False_False_False_True_False_True-pool_type-None_None_None_maxpool_None_maxpool-cifar100.yaml
python3 train.py --config ./config/cifar100/218-stage-2_4_6_1_1-ratio-0.75_0.75_1.0_0.25_0.5-op-vgg_vgg_repvgg_vgg_vgg-pool-False_False_False_False_True_True-pool_type-None_None_None_None_avgpool_maxpool-cifar100.yaml
python3 train.py --config ./config/cifar100/219-stage-2_3_1_6_1-ratio-0.125_0.125_0.5_0.625_0.25-op-repvgg_repvgg_repvgg_repvgg_repvgg-pool-False_False_False_False_True_True-pool_type-None_None_None_None_avgpool_avgpool-cifar100.yaml