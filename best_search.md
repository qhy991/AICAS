|NO.|layer|ratio|op|pool|pool_type|datset|flops|params|Acc|constrain|
|-|-|-|-|-|-|-|-|-|-|-|
|0|2-2-4-6-8|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|123.53|4.15||acc+params+|
|1|2-2-4-3-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|94.13|2.12||acc+flops|
|3|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76||acc+flops+params||
|4|2-2-4-6-5|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|123.08|4.04||acc+flops+params||
|5|2-2-4-6-8|1.0-1.0-1.0-1.0-1.0|repvgg-repvgg-vgg-vgg-vgg|False-False-True-True-True|None-None-avgpool-avgpool-avgpool-maxpool|cifar10|214.41|21.59||acc+flops+params||