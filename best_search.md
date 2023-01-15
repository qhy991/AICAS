|NO.|layer|ratio|op|pool|pool_type|datset|flops|params|Acc|constrain|
|-|-|-|-|-|-|-|-|-|-|-|
|0|2-2-4-6-8|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|123.53|4.15|90.74|acc+params+|
|1|91.89|
|1|2-2-4-3-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|94.13|2.12|85.96|acc+flops|
|3|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76|93.03|acc+flops+params|
|4|2-2-4-6-5|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|123.08|4.04||acc+flops+params|
|5|2-2-4-6-8|1.0-1.0-1.0-1.0-1.0|repvgg-repvgg-vgg-vgg-vgg|False-False-True-True-True|None-None-avgpool-avgpool-avgpool-maxpool|cifar10|214.41|21.59||acc+flops+params||
|6|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76|93.16(chaneg lr + 150 epoch)|acc+flops+params|
|7|2-2-4-1-1|0.75-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|57.97|0.92|91.5(change lr)|acc+flops+params|
|8|2-2-4-1-1|0.75-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|57.97|0.92|92.16(change lr)|acc+flops+params|
|9|2-1-4-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|75.07|0.94|91.49(change lr)|acc+flops+params|
|10|2-1-3-1-1|1.0-0.125-1.0-1.0-1.0|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|69.74|1.83|91.27(change lr)|acc+flops+params|
|11|2-1-4-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|75.07|0.94|92.11(150epoch)(change lr)|acc+flops+params|
|12|2-1-3-1-1|1.0-0.125-1.0-1.0-1.0|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|69.74|1.83|91.71(150epoch)(change lr)|acc+flops+params|
|13|2-1-2-2-1|1.0-0.125-0.125-1.0-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|51.58|0.82|89.29|acc+flops+params|
|14|2-1-2-2-1|1.0-0.125-0.125-1.0-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|51.58|0.82|89.41(change lr)|acc+flops+params|
|15|2-1-4-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|75.07|0.94|91.49|acc+flops+params|
|16|2-1-3-1-2|0.75-0.125-0.125-0.125-1.0|vgg-vgg-vgg-vgg-repvgg|False-False-False-True-False|None-None-None-avgpool-None-None|cifar10|34.34|2.55||acc+flops+params|
|17|1-3-1-1-1|1.0-0.125-1.0-1.0-0.125|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.48|0.46|87.12|acc+flops+params|
|18|2-1-3-8-1|0.25-0.875-0.125-0.125-0.625|vgg-vgg-vgg-vgg-vgg|False-False-False-True-False|None-None-None-maxpool-None-None|cifar10|7.38|0.19|83.79|acc+flops+params|
|19|1-1-1-2-1|1.0-0.125-0.25-0.125-0.625|repvgg-repvgg-vgg-vgg-repvgg|False-False-False-False-True|None-None-None-None-maxpool-None|cifar10|3.78|0.12|83.69|acc+flops+params|
|20|1-1-1-1-1|0.75-0.25-0.75-0.125-0.375|vgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|4.89|0.11|85.04|acc+flops+params|
|21|1-3-1-2-1|0.75-0.125-0.125-0.125-1.0|vgg-vgg-vgg-vgg-repvgg|False-False-False-True-False|None-None-None-avgpool-None-None|cifar10|3.63|0.17|79.22|acc+flops+params|
|22|1-3-1-1-1|1.0-0.125-1.0-1.0-0.125|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.48|0.46|87.12|acc+flops+params|
|23|1-1-1-1-1|0.75-0.375-1.0-0.125-0.875|vgg-vgg-repvgg-vgg-vgg|False-False-True-False-True|None-None-maxpool-None-avgpool-None|cifar10|7.09|0.21|83.02|acc+flops+params|
|24|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76|92.84(change lr min_lr)92.84|acc+flops+params|
|25|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76|(change lr )92.83|acc+flops+params|
|26|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76|(change lr 150epoch )92.88|acc+flops+params|
|27|2-2-1-5-4|0.75-0.75-0.625-0.125-0.125|vgg-vgg-vgg-vgg-repvgg|False-False-False-False-True|None-None-None-None-maxpool-None|cifar10|37.39|0.29|90.26|acc+flops+params|
|28|1-4-1-1-1|0.5-0.125-0.125-0.25-0.875|repvgg-repvgg-repvgg-repvgg-vgg|False-False-True-False-False|None-None-avgpool-None-None-None|cifar10|3.18|0.28|83.9|acc+flops+params|
|29|1-1-1-2-1|1.0-0.5-0.75-0.125-1.0|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-True|None-None-None-None-maxpool-None|cifar10|9.51|0.24|87.02|acc+flops+params|
|30|1-3-1-1-3|1.0-0.5-0.5-0.5-0.125|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|14.54|0.28|89.2|acc+flops+params|
|31|2-1-2-7-4|1.0-1.0-0.625-0.125-0.125|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|58.01|0.39|90.35|acc+flops+params|
|32|2-1-1-1-1|0.375-0.75-0.75-0.375-0.375|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|13.36|0.31|89.37|acc+flops+params|
|33|1-4-1-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.37|0.46|87.96|acc+flops+params|
|34|1-1-1-1-1|0.5-0.5-1.0-0.375-0.625|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|8.56|0.44|88.49|acc+flops+params|
|35|1-4-1-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.37|0.46|87.96|acc+flops+params|
|36|1-4-1-1-4|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.82|0.57|87.48|acc+flops+params|
|37|1-3-1-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.22|0.46|87.40|acc+flops+params|
|38|1-4-1-1-2|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.52|0.5|87.32|acc+flops+params|
|39|2-4-1-1-3|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|47.42|0.57|88.95|acc+flops+params|
|40|1-1-1-1-1|0.875-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|8.54|0.46|87.5|acc+flops+params|
|41|2-4-1-1-1|0.875-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|37.91|0.49|89.03|acc+flops+params|
|42|2-4-1-1-4|0.875-0.25-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-avgpool|cifar10|41.33|0.62|90.14|acc+flops+params|
|46|2-4-1-1-4|0.875-0.25-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-True|None-None-None-None-avgpool-None|cifar10|41.33|0.62|89.93|acc+flops+params|
|43|2-4-1-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|47.12|0.5|88.75|acc+flops+params|
|44|1-1-1-2-1|0.5-0.5-1.0-0.125-0.375|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|6.77|0.15|86.65|acc+flops+params|
|45|2-2-1-1-1|0.125-0.125-1.0-0.5-0.125|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|4.47|0.23|86.26|acc+flops+params|
|vgg11|1-1-2-2-2|1.0-1.0-1.0-1.0-1.0|vgg-vgg-vgg-vgg-vgg|False-False-True-True-True|None-None-maxpool-maxpool-maxpool-maxpool|cifar10|54.12|4.69||acc+flops+params|
|vgg13|2-2-2-2-2|1.0-1.0-1.0-1.0-1.0|vgg-vgg-vgg-vgg-vgg|False-False-True-True-True|None-None-maxpool-maxpool-maxpool-maxpool|cifar10|101.64|4.77||acc+flops+params|
|vgg16|2-2-3-3-3|1.0-1.0-1.0-1.0-1.0|vgg-vgg-vgg-vgg-vgg|False-True-True-True-True|None-maxpool-maxpool-maxpool-maxpool-maxpool|cifar10|130.0|7.87||acc+flops+params|