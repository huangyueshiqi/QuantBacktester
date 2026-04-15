# QuantBacktester - 量化回测系统

## 项目简介

QuantBacktester是一个功能完善的量化投资回测系统，基于Python和backtrader框架开发。该系统支持多种回测策略、分红处理、多资产配置以及增量回测等高级功能，为量化交易策略的开发和测试提供了强大的工具。

## 功能特点

- **Alpha选股策略**：基于预测分数(alpha)选择股票并进行交易
- **指数策略**：支持基于指数的投资策略回测和调仓计划
- **多资产配置**：支持使用cvxopt进行资产优化配置
- **分红处理**：自动处理股票分红、送股、拆股等公司行为
- **ST股票处理**：自动剔除ST股票
- **增量回测**：支持增量回测，可以从之前的状态继续回测
- **调仓计划**：支持根据预先设定的调仓计划进行策略回测
- **等权重策略**：支持使用等权重进行资产配置
- **丰富的分析器**：包含收益率、夏普比率、回撤等多种分析指标

## 项目结构

```
QuantBacktester/
├── core/                 # 核心功能模块
│   ├── analyzer.py       # 策略分析器
│   ├── benchmark_calculator.py  # 基准计算器
│   ├── dividend_handler.py      # 分红处理器
│   ├── portfolio.py      # 投资组合管理
│   ├── state_manager.py  # 状态管理器----用于增量回测
│   ├── strategy.py       # 策略定义
│   └── trade_executor.py # 交易执行器
├── data/                 # 数据处理模块
│   ├── loader.py         # 数据加载器
│   └── models.py         # 数据模型定义
├── input/                # 输入数据目录
├── plot/                 # 图表输出目录
├── result/               # 结果输出目录
├── state/                # 状态保存目录
├── utils/                # 工具函数
│   ├── config.py         # 配置管理
│   └── helpers.py        # 辅助函数
└── main.py               # 主程序入口
```

## 目录
`plot`目录当前未使用，
`result`目录，`state`目录在增量回测时被使用，
`input`目录在基于调仓表文件回测和基于指数调仓表文件回测时被使用


## 环境要求

1. 57服务器上的zcenv
2. 可能遇到的问题
如果遇到cx_Oracle.DatabaseError:DPI-1047错误
```运行时环境变量需要加上
LD_LIBRARY_PATH=/oracle/client:/home/quant/oracle ORACLE_HOME=/oracle/client
```

## 快速开始

### 执行回测

```
python main.py
```

### 参数说明

- `--mode`：指定四种回测模式。['full', 'incremental','rebalanceTable','index_rebalanceTable']
- `--config`：指定配置文件路径,当前并没有，默认为空
- `--predict`：预测数据文件路径,基于score的凸优化
- `--pool`：指定股票池文件路径
- `--rebalance_file`：股票调仓表文件路径
- `--index_rebalance_file`：指数调仓表文件路径
- `--output`：程序运行命令行结果文件路径.如果设置为空,那么程序运行过程将在命令行输出.考虑到长时间回测程序可能输出过多命令行信息,建议放在output文件中查看
- `--verbose`：输出详细日志

对应回测模式中的其它特有参数可以进入main文件中的对应方法中调整。一般来说调整的都是回测结束日期，方便调试代码。其它参数可以在config文件中调整。

## 配置文件

系统支持通过JSON格式的配置文件对回测设置进行详细配置，主要配置项包括：

- 数据库连接信息
- 回测参数（初始资金、手续费、滑点等）
- 策略参数（周期、执行价格、阈值等）
- 凸优化时优化器参数
- 基准指数设置，默认为中证800指数
- 文件路径配置


## 数据源

系统支持多种数据源：

1. **Oracle数据库**：通过cx_Oracle连接Oracle数据库获取股票数据
2. **CSV文件**：支持从CSV文件读取股票数据和预测分数
3. **调仓计划**：支持从CSV文件读取调仓计划数据

## 主要策略

### AlphaStrategy

支持基于预测分数选择股票并进行交易的策略和基于调仓表选择股票并进行交易的策略。主要参数：

- `period`：默认周期
- `execution_price`：执行价格类型（如open, close等）
- `score_threshold`：分数阈值
- `topk`：选股数量
- `low_stopping`：跌停判断阈值
- `high_limit`：涨停判断阈值
- `volPercent`：成交量比例阈值
- `max_position`：最大持仓比例
- `dealDivident`：是否处理分红
- `cash2shares`：现金分红是否填权

### IndexStrategy

基于指数成分股进行交易的策略，支持根据调仓计划进行操作。主要参数：

- `rebalance_plan`：调仓计划数据


## 基于调仓表进行回测，与策略选股对接

基于调仓表进行交易的策略，支持根据调仓计划进行操作。
main.py中需要的参数
- `mode`："rebalanceTable"
- `rebalance_file`：股票调仓表文件路径
传入的调仓表必须包含：['date','stock','weight']三列,后两列列名可以不一样


## 增量回测

系统支持增量回测功能，相关参数：

- `save_state`：是否保存回测状态
- `month_ibt`：是否按月进行增量回测
这两个参数需要同时关闭同时打开，都为False时关闭
增量回测可以显著提高回测效率，尤其适用于长期策略的持续评估。

## 结果分析

系统输出的分析结果包括：

- 年化收益率
- 夏普比率
- 最大回撤
- 交易次数和胜率----增量回测时可能计算有问题
- 盈亏比----增量回测时可能计算有问题
- 收益曲线图表
等等指标
