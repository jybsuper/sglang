# PDL control matrix summary

Negative percentages mean architecture-auto PDL is faster than forced-off. Each row combines forward and reverse process order.

| GPU | T | Site | Execution | Auto us | Off us | Auto vs off median [min, max] | Stable sign |
|:---|---:|:---|:---|---:|---:|:---|:---:|
| NVIDIA GB300 | 32 | down | cuda_graph | 6.035 | 6.344 | -4.87% [-6.35, -3.38] | True |
| NVIDIA GB300 | 32 | down | eager | 19.248 | 17.987 | +7.03% [+5.16, +8.90] | True |
| NVIDIA GB300 | 32 | gate | cuda_graph | 26.005 | 26.211 | -0.79% [-0.81, -0.77] | True |
| NVIDIA GB300 | 32 | gate | eager | 26.306 | 28.768 | -8.56% [-8.56, -8.56] | True |
| NVIDIA GB300 | 2048 | down | cuda_graph | 17.814 | 18.016 | -1.12% [-1.13, -1.12] | True |
| NVIDIA GB300 | 2048 | down | eager | 19.125 | 20.578 | -7.06% [-7.63, -6.49] | True |
| NVIDIA GB300 | 2048 | gate | cuda_graph | 116.651 | 116.574 | +0.07% [+0.01, +0.13] | True |
| NVIDIA GB300 | 2048 | gate | eager | 116.984 | 118.652 | -1.41% [-1.44, -1.38] | True |
| NVIDIA H200 | 32 | down | cuda_graph | 6.736 | 7.024 | -4.10% [-4.17, -4.03] | True |
| NVIDIA H200 | 32 | down | eager | 19.186 | 18.930 | +1.35% [+0.79, +1.91] | True |
| NVIDIA H200 | 32 | gate | cuda_graph | 35.738 | 36.050 | -0.87% [-0.87, -0.86] | True |
| NVIDIA H200 | 32 | gate | eager | 35.927 | 37.671 | -4.63% [-4.64, -4.62] | True |
| NVIDIA H200 | 2048 | down | cuda_graph | 26.620 | 26.898 | -1.03% [-1.15, -0.91] | True |
| NVIDIA H200 | 2048 | down | eager | 26.822 | 28.331 | -5.33% [-5.43, -5.23] | True |
| NVIDIA H200 | 2048 | gate | cuda_graph | 136.041 | 136.266 | -0.16% [-0.19, -0.14] | True |
| NVIDIA H200 | 2048 | gate | eager | 136.130 | 137.784 | -1.20% [-1.25, -1.15] | True |
