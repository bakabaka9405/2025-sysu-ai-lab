# 实验二 一阶逻辑归结算法的实现

## 项目结构

```
├── main.py               # 主程序
├── knowledge_base.py     # 知识库，子句，谓词等数据结构模块
├── resolution.py         # 归结算法模块
└── config.py             # 配置文件
```

## 环境要求

- Python 3.9 或更高版本

## 使用说明

执行主程序：
```powershell
python .\main.py
```

程序启动时不会有任何输出，此时在等待控制台输入。输入格式为标准的子句集，如：

```
KB = {(GradStudent(sue),),(~GradStudent(x),Student(x)),(~Student(x),HardWorker(x)),(~HardWorker(sue),)}
```

输入后回车，程序会自动进行推断并输出结果。

要启用/关闭部分功能，参见`config.py`

**注意** 只为输入的变量格式设置了两种形式：单字符变量和双字符变量，二者只可选其一。具体信息参见`config.py`的`variable_name_type`字段。如果输入格式与设定的字段格式不匹配，很可能得到错误的结果。