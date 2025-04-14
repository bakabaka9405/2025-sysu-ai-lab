# 实验三 搜索算法（任务二 启发式搜索 15-puzzle）

## 项目结构

```
├── src/                              # 源代码目录
│   ├── main.py                       # 主程序
│   ├── GeneticAlgTSP.py              # TSP 算法封装模块
│   ├── worker.py                     # 并行计算模块
│   ├── selection.py                  # 选择算法模块
│   ├── crossover.py                  # 交叉算法模块
│   ├── mutation.py                   # 变异算法模块
│   ├── fitness_transform.py          # 适应度转换模块
│   ├── util.py                       # 实用函数
│   └── config.py                     # 配置文件
├── data/                             # 数据目录
└── README.md                         # 项目说明文件
```

## 环境要求

- Python 3.9 或更高版本

## 使用说明

修改`config.py`指定输入文件后，执行主程序：
```powershell
python .\main.py
```

或根据实验指导书要求直接实例化 GeneticAlgTSP 类：
```python
from GeneticAlgTSP import GeneticAlgTSP

ga = GeneticAlgTSP.GeneticAlgTSP('/path/to/data.txt')

res = ga.iterate(100)

print(res)
```

和以往实验代码相同，要修改运行配置，参见`config.py`。