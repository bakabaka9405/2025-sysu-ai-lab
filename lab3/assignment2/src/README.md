# 实验三 搜索算法（任务二 启发式搜索 15-puzzle）

## 项目结构

```
├── main.py               # 主程序
├── a_star.py             # A* 算法模块
├── heuristic.py          # 启发式函数模块
├── util.py               # 实用函数
└── config.py             # 配置文件
```

## 环境要求

- Python 3.9 或更高版本

## 使用说明

执行主程序：
```powershell
python .\main.py
```

输入格式有四种，参见`config.py`的`a_star_initial_state`

简单且用得上的两种：

要从控制台输入，将`a_star_initial_state`设为 None，控制台输入格式形如
```
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 0
```
否则，`a_star_initial_state`若为一个二维数组，则不需要从控制台输入，直接将其作为参数传入 A* 算法