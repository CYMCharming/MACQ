# MACQ

MACQ 是面向 Jev 类决策模型的训练后量化方法。它保留原生问题编译与决策读出，用前两名候选的间隔梯度对校准 token 采样，再根据权重与激活共同量化后的输出误差选择配置。对激活量化误差拟合的补偿会在量化前限幅并折入权重。

[安装与复现](docs/reproduce-current.md) · [实现细节](docs/method.md) · [主实验表](docs/results.md)

命令 `macq optimize` 拟合并保存 W4A4 量化结果，每层使用 16,384 个校准 token。主实验覆盖 Kev、Intern-Decision、StartLux-Decision 和 Laya 共 11 个模型，每个模型在 22 个数据集上等权平均。MACQ 的跨模型平均准确率为 **74.82%**，原生 BF16 为 77.44%。[逐模型和逐数据集结果](docs/results.md)
