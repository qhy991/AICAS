|NO.|layer|ratio|op|pool|pool_type|datset|flops|params|Acc|constrain|
|-|-|-|-|-|-|-|-|-|-|-|
|0|2-2-4-6-8|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|123.53|4.15|90.74|acc+params+|
|1|2-2-4-3-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|94.13|2.12|85.96|acc+flops|
|3|2-2-4-1-2|0.875-1.0-1.0-1.0-0.875|repvgg-vgg-vgg-vgg-vgg|False-False-True-True-False|None-None-avgpool-avgpool-None-None|cifar10|97.55|3.76|93.03|acc+flops+params|
|4|2-2-4-6-5|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|123.08|4.04||acc+flops+params|
|5|2-2-4-6-8|1.0-1.0-1.0-1.0-1.0|repvgg-repvgg-vgg-vgg-vgg|False-False-True-True-True|None-None-avgpool-avgpool-avgpool-maxpool|cifar10|214.41|21.59||acc+flops+params|
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
|47|1-4-1-1-1|1.0-0.125-1.0-1.0-0.25|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.97|0.61|88.26|acc+flops+params|
|48|1-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|21.35|0.56||acc+flops+params|
|49|1-1-1-1-1|1.0-1.0-1.0-0.875-0.125|repvgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|20.67|0.5|89.91|acc+flops+params|
|50|1-4-1-1-1|1.0-0.125-1.0-1.0-0.125|vgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.64|0.46|86.25|acc+flops+params|
|51|1-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|21.3|0.56|89.34|acc+flops+params|
|52|2-1-1-1-1|0.125-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|12.13|0.52|88.87|acc+flops+params|
|53|1-1-1-1-1|1.0-1.0-1.0-1.0-1.0|repvgg-repvgg-vgg-vgg-repvgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|25.42|1.59|89.84|acc+flops+params|
|54|2-1-1-1-1|0.125-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|12.13|0.52|88.87|acc+flops+params|
|55|2-1-1-1-1|0.125-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|12.13|0.52|88.87|acc+flops+params|
|56|1-3-1-1-1|1.0-0.125-1.0-1.0-0.125|repvgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|9.2|0.46|87.83|acc+flops+params|
|57|1-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-repvgg-repvgg-repvgg|False-False-False-False-False|None-None-None-None-None-maxpool|cifar10|21.23|0.56|85.76|acc+flops+params|
|58|1-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-repvgg-repvgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|21.33|0.56|89.54|acc+flops+params|
|59|1-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-vgg-repvgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|21.28|0.56|89.89|acc+flops+params|
|60|1-2-1-1-1|1.0-0.25-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|11.29|0.48|88.39|acc+flops+params|
|61|1-1-1-1-1|1.0-0.875-1.0-1.0-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|19.52|0.54|89.45|acc+flops+params|
|62|2-1-1-1-1|0.125-1.0-1.0-1.0-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|12.1|0.52|89.16|acc+flops+params|
|63|1-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|21.28|0.56|89.19|acc+flops+params|
|64|1-1-1-1-1|1.0-1.0-1.0-0.75-0.125|repvgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|20.01|0.45|89.02|acc+flops+params|
|65|2-4-4-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|116.0|1.15|93.41|acc+flops+params|
|66|2-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|59.08|0.59|90.57|acc+flops+params|
|67|2-4-1-1-4|1.0-0.25-1.0-1.0-0.125|repvgg-vgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|50.68|0.63|90.12|acc+flops+params|
|68|2-4-3-1-4|1.0-0.875-1.0-1.0-0.125|repvgg-repvgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|98.33|1.07|92.65|acc+flops+params|
|69|2-4-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|87.57|0.7|92.27|acc+flops+params|
|70|2-4-6-1-1|1.0-0.875-1.0-0.75-0.125|repvgg-repvgg-vgg-vgg-vgg|False-True-False-True-False|None-maxpool-None-maxpool-None-None|cifar10|124.96|1.29|92.79|acc+flops+params|
|71|1-1-1-1-1|0.375-1.0-1.0-1.0-0.5|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|16.12|0.98|89.53|acc+flops+params|
|72|2-4-1-1-2|1.0-0.5-1.0-1.0-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|59.26|0.6|91.58|acc+flops+params|
|73|2-3-1-1-1|1.0-0.5-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|56.77|0.56|90.98|acc+flops+params|
|74|2-4-1-1-1|1.0-0.75-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|71.62|0.63|92.03|acc+flops+params|
|75|1-1-1-1-1|0.875-1.0-1.0-1.0-0.125|repvgg-repvgg-repvgg-repvgg-repvgg|False-False-False-False-False|None-None-None-None-None-maxpool|cifar10|19.83|0.55|76.65|acc+flops+params|
|76|2-4-2-1-4|1.0-0.625-0.75-1.0-0.125|repvgg-vgg-vgg-vgg-repvgg|False-True-False-False-False|None-maxpool-None-None-None-maxpool|cifar10|68.8|0.7|91.35|acc+flops+params|
|77|1-1-1-1-1|0.25-1.0-1.0-1.0-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|12.9|0.53|88.74|acc+flops+params|
|78|2-4-1-1-4|0.875-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-vgg-repvgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|77.54|0.8|91.63|acc+flops+params|
|79|2-1-1-1-1|0.5-1.0-1.0-1.0-0.125|repvgg-vgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|25.18|0.55|90.55|acc+flops+params|
|80|2-4-5-1-1|1.0-0.5-1.0-1.0-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|96.86|1.16|92.27|acc+flops+params|
|81|2-4-3-1-1|0.125-1.0-1.0-1.0-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|59.48|0.93|91.6|acc+flops+params|
|82|2-4-1-1-4|0.875-0.75-1.0-1.0-0.125|repvgg-repvgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|61.91|0.73|91.81|acc+flops+params|
|83|2-4-1-1-1|1.0-1.0-1.0-0.625-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-False-False-True-False|None-None-None-maxpool-None-None|cifar10|85.57|0.54|92.87|acc+flops+params|
|84|2-4-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-vgg-repvgg-repvgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|87.56|0.7|91.94|acc+flops+params|
|85|1-1-1-1-1|1.0-1.0-1.0-0.25-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|17.32|0.22||acc+latency|
|86|1-1-1-1-1|1.0-1.0-1.0-0.875-0.125|repvgg-vgg-repvgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|20.65|0.5|89.59|acc+latency|
|87|2-1-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|59.02|0.59|90.60|acc+latency|
|88|1-1-1-1-1|0.25-1.0-0.25-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|5.76|0.25|87.51|acc+latency|
|89|2-3-3-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|96.83|0.96|93.14|acc+latency|
|90|2-2-2-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|77.92|0.78|92.31|acc+latency|
|91|2-2-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|68.45|0.63|91.39|acc+latency|
|92|2-3-4-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|106.3|1.11|92.8|acc+latency|
|93|2-2-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|68.45|0.63|91.99|acc+latency|
|94|2-3-3-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|96.83|0.96|93.14|acc+latency|
|95|2-1-3-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|77.96|0.89|92.58|acc+latency|
|96|1-2-1-1-1|1.0-1.0-1.0-1.0-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|30.71|0.59|90.72|acc+latency|
|97|1-1-1-1-1|0.25-1.0-1.0-0.75-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-avgpool|cifar10|11.54|0.42|88.24|acc+latency|
|98|1-1-1-1-1|1.0-1.0-1.0-0.5-0.25|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|18.91|0.41|89.45|acc+latency|
|99|1-1-1-1-1|0.25-1.0-0.25-0.875-0.125|repvgg-repvgg-vgg-vgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|5.55|0.22|87.43|acc+latency|
|100|1-1-1-1-1|0.25-1.0-1.0-0.625-0.25|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|11.24|0.45|88.82|acc+latency|
|101|1-1-1-1-1|0.25-1.0-0.25-0.75-0.125|repvgg-repvgg-vgg-repvgg-vgg|False-False-False-False-False|None-None-None-None-None-None|cifar10|5.32|0.2|87.16|acc+latency|
|102|2-4-3-1-1|1.0-1.0-0.625-1.0-0.375|repvgg-repvgg-vgg-vgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-avgpool|cifar10|92.39|0.98|92.29|acc+latency|
|103|2-4-1-1-1|1.0-0.875-0.75-0.875-0.125|vgg-vgg-vgg-vgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-None|cifar10|76.96|0.53|91.21|acc+latency|
|104|2-4-3-1-2|1.0-0.875-0.625-1.0-0.125|repvgg-repvgg-vgg-vgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-avgpool|cifar10|83.17|0.68|92.26|acc+latency|
|105|2-4-3-1-2|1.0-0.875-0.625-1.0-0.125|repvgg-repvgg-vgg-vgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-avgpool|cifar10|83.17|0.68|92.26|acc+latency|
|106|2-4-3-1-1|1.0-1.0-0.75-1.0-0.25|repvgg-vgg-vgg-vgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-avgpool|cifar10|96.5|0.93|92.24|acc+latency|
|107|1-1-1-1-1|0.125-0.5-0.125-0.875-0.25|vgg-vgg-vgg-vgg-vgg|False-True-False-False-False|None-avgpool-None-None-None-None|cifar10|2.74|0.3|84.62|acc+latency|
|108|2-4-2-1-1|1.0-1.0-0.625-1.0-0.375|repvgg-vgg-vgg-vgg-vgg|False-True-False-False-False|None-maxpool-None-None-None-avgpool|cifar10|88.95|0.92|92.11|acc+latency|
|vgg11|1-1-2-2-2|1.0-1.0-1.0-1.0-1.0|vgg-vgg-vgg-vgg-vgg|False-False-True-True-True|None-None-maxpool-maxpool-maxpool-maxpool|cifar10|54.12|4.69|91.23|acc+flops+params|
|vgg13|2-2-2-2-2|1.0-1.0-1.0-1.0-1.0|vgg-vgg-vgg-vgg-vgg|False-False-True-True-True|None-None-maxpool-maxpool-maxpool-maxpool|cifar10|101.64|4.77|93.24|acc+flops+params|
|vgg16|2-2-3-3-3|1.0-1.0-1.0-1.0-1.0|vgg-vgg-vgg-vgg-vgg|False-True-True-True-True|None-maxpool-maxpool-maxpool-maxpool-maxpool|cifar10|130.0|7.87|92.52|acc+flops+params|
