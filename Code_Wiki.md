# QuantBacktester - 量化回测系统 Code Wiki

## 1. 项目整体架构

QuantBacktester 是一个基于 **Python** 和 **Backtrader** 框架开发的功能完善的量化投资回测系统。该系统设计采用了高度模块化的架构，将数据加载、策略定义、交易执行、状态管理、约束检查等逻辑解耦，遵循单一职责原则。

系统支持多种高级量化交易场景：
- **多种策略模式**：支持 Alpha 选股（基于预测模型如 GRU 生成的 score）、TopK 策略、基于 LLM 信号的交易、指数策略以及基于自定义调仓表的交易。
- **增量回测支持**：支持通过 `StateManager` 保存每日资金、持仓及收益流水，实现断点续传（增量回测），极大提升了长周期回测的开发调试效率。
- **复杂交易行为模拟**：实现了真实的 A 股交易特性，包括 ST 股票处理、分红派息扣税、送股拆股除权、涨跌停板无法成交判断等。
- **优化与约束控制**：内置了 `cvxopt` 凸优化进行资产权重分配，并可通过 `ConstraintManager` 进行持股周期、市值、连续涨跌停等严格的交易约束限制。

---

## 2. 目录结构与模块职责

```text
/workspace/
├── core/                   # 核心回测逻辑与组件模块 (重点)
│   ├── analyzer.py         # 自定义策略分析器（夏普、回撤、收益率等计算）
│   ├── benchmark_calculator.py # 基准收益计算器（如中证800指数对比）
│   ├── buy_strategy.py     # 纯买入列表策略实现
│   ├── constraint_manager.py # 交易约束管理器（持仓期、市值、涨跌停限制等）
│   ├── dividend_handler.py # 分红与除权除息处理器
│   ├── index_strategy.py   # 指数标的交易策略
│   ├── llm_strategy.py     # 接入 LLM 预测信号的交易策略
│   ├── portfolio.py        # 资金与持仓管理器封装
│   ├── position.py         # 头寸追踪模块
│   ├── state_manager.py    # 增量回测状态保存与恢复
│   ├── strategy.py         # 核心 AlphaStrategy 策略基类
│   ├── topk_strategy.py    # TopK 选股策略
│   └── trade_executor.py   # 交易指令执行器（计算买卖数量、下发Order）
├── data/                   # 数据接入与预处理模块
│   └── loader.py           # 包含 OracleDataLoader、StockDataService 及预处理类
├── utils/                  # 通用工具与配置
│   ├── config.py           # 全局配置管理（读取配置参数）
│   ├── helpers.py          # 辅助函数（如 cvxopt 优化、调仓日计算、绘图等）
│   └── util.py             # 杂项工具函数
├── llm/                    # 存放与 LLM（大模型）相关的策略预处理和笔记本
├── xqb/                    # 实验性质文件与探索 Notebooks
├── main.py                 # 主程序统一入口（CLI 命令行解析与任务分发）
├── run_topk.py             # 特化的 TopK 模式快捷执行脚本
├── pred_score.py           # 模型预测打分生成脚本
└── README.md               # 项目基础说明文档
```

---

## 3. 关键类与函数说明

### 3.1 调度层 (`main.py`)
- **`BacktestManager`**
  - **职责**：回测调度中心，连接数据层与 Backtrader 引擎。
  - **核心方法**：
    - `_prepare_backtest_data()`: 调用数据服务获取并预处理所有股票、分红、ST等回测数据。
    - `_setup_cerebro()`: 初始化 Cerebro，设置初始资金、佣金、滑点并挂载 Analyzers。
    - `run_full_backtest() / run_incremental_backtest() / run_topk_backtest()`: 针对不同 `--mode` 执行相应回测流程。

### 3.2 数据层 (`data/loader.py`)
- **`OracleDataLoader`**
  - **职责**：封装 cx_Oracle 数据库连接，执行 SQL 提取股票明细、ST数据、分红送转数据、日历等。
- **`StockDataService`**
  - **职责**：数据门面（Facade）。支持通过数据库（OracleDataLoader）或本地 Qlib 数据源拉取所需行情。
  - **核心方法**：`get_stock_data_for_backtest()` 统一获取回测所需的标的行情与基准数据。
- **`DataPreprocessor`**
  - **职责**：清洗整理数据结构，对接 Backtrader。
  - **核心方法**：`preprocess_stock_detail_data()` 合并打分模型 (score) 数据与行情；`check_for_dividends_from_df()` 处理除权除息日与分红现金/送股比例。

### 3.3 策略与执行层 (`core/`)
- **`AlphaStrategy`** (`strategy.py`)
  - **职责**：继承自 `bt.Strategy` 的主策略逻辑。
  - **核心生命周期**：
    - `__init__()`: 初始化并实例化子组件 (`TradeExecutor`, `DividendHandler`, `ConstraintManager`, `Portfolio`)。
    - `next()`: Bar 到达时的核心逻辑。依次执行：恢复增量状态 -> 剔除最新 ST 股票 -> 处理当日分红 -> 判断是否为调仓日 -> 调用 cvxopt 计算目标权重 -> 调用 Executor 执行调仓。
    - `notify_order()` / `notify_trade()`: 订单回调处理，计算手续费、扣税及交易胜率。
- **`TradeExecutor`** (`trade_executor.py`)
  - **职责**：具体执行买卖的组件，与 Broker 交互。
  - **核心方法**：
    - `rebalance_portfolio()`: 对比目标权重与历史权重，先执行卖出（释放资金），再执行买入。
    - `to_buy()` / `to_sell()`: 计算实际需买卖的数量，校验涨跌停板限制（`high_limit` / `low_stopping`）后下发订单。
- **`ConstraintManager`** (`constraint_manager.py`)
  - **职责**：约束校验中心。在调仓前，根据配置决定某只股票是否可买/必须保留。支持的约束包含：`MinHoldingDays`（最小持有期）、市值过滤、连续涨跌停过滤等。
- **`DividendHandler`** (`dividend_handler.py`)
  - **职责**：在除权除息日对持仓股票派发现金、送股，并进行红利税扣除。
- **`StateManager`** (`state_manager.py`)
  - **职责**：处理断点续传逻辑，通过 Pickle 序列化或反序列化资金、头寸记录。

### 3.4 工具层 (`utils/`)
- **`cvxopt`** (`helpers.py`)
  - **职责**：使用凸优化（Convex Optimization）算法，基于基准指数成分权重和股票打分，计算调仓日的最优投资组合资产权重配置。
- **`PandasDataExtend`** (`main.py` 内部)
  - **职责**：继承自 `bt.feeds.PandasData`，扩展 Backtrader 的 Line 数据流，增加 `score`, `limit`, `stopping`, `tradestatus` 等自定义列供策略使用。

---

## 4. 依赖关系与数据流

1. **预测流（外部依赖）**：外部机器学习/深度学习模型（如 GRU）生成带有 `score` 列的预测文件（CSV）。
2. **数据流**：
   - 预测文件 CSV + 数据库（或 Qlib）拉取的行情数据 -> `DataPreprocessor` 合并为完整的 DataFrame。
   - DataFrame 被包装成 `PandasDataExtend` 注入到 `Cerebro`。
3. **控制流**（每到达一个交易日 Bar）：
   - `Cerebro` 触发 `Strategy.next()`。
   - 检查 `DividendHandler`，执行资金账户红利派发。
   - 若命中 `rebalancing_days`，进入调仓流程：
     - `ConstraintManager` 过滤掉不合规的股票（如 ST、涨停、未满足持股期）。
     - 调用 `cvxopt` 计算最终的目标权重组合。
     - 交给 `TradeExecutor.rebalance_portfolio()` 统一下单。
   - `Cerebro` 下一 Bar 进行撮合成交，回调 `notify_order` 和 `notify_trade`。

---

## 5. 项目运行方式

### 5.1 环境准备
建议在 Linux 环境中运行，如果使用 Oracle 数据库遇到 `DPI-1047` 错误，需配置环境变量：
```bash
export LD_LIBRARY_PATH=/oracle/client:/home/quant/oracle
export ORACLE_HOME=/oracle/client
```
核心依赖：`backtrader`, `pandas`, `numpy`, `cx_Oracle`, `matplotlib`, `qlib`。

### 5.2 启动命令与参数
系统统一通过 `main.py` 入口运行，使用 `--mode` 区分不同的回测任务。

**全量选股回测**：
```bash
python main.py --mode full --predict input/prediction_results.csv --pool input/section_codes.csv
```

**TopK 选股回测**：
```bash
python main.py --mode topk --predict input/prediction_results.csv --pool input/section_codes.csv
```

**增量回测**（适用于每日盘后执行，继承历史状态）：
```bash
python main.py --mode incremental --predict input/prediction_results.csv
```

**基于调仓表的回测**（不依赖预测模型，直接读取权重表）：
```bash
python main.py --mode rebalanceTable --rebalance_file ./input/trade_log.csv
```

**核心参数说明**：
- `--mode`: 运行模式，可选 `['full', 'topk', 'incremental', 'rebalanceTable', 'index_rebalanceTable', 'buy', 'llm']`。
- `--config`: 配置文件路径，可覆盖策略参数。
- `--predict`: Alpha 预测分数 CSV 文件路径。
- `--pool`: 候选股票池代码 CSV。
- `--output`: 标准输出重定向的文件路径，避免控制台日志过多。
- `--plot_output`: 回测结果收益曲线图的保存路径。
- `--verbose`: 是否开启详细的 Backtrader 买卖点日志。

### 5.3 结果产出
回测结束后，系统将会在控制台（或 `--output` 指定的文件）中打印核心绩效指标：
- **Annual Return**（年化收益率）
- **Sharpe Ratio**（夏普比率）
- **Max DrawDown**（最大回撤）
- **自定义指标**（胜率、盈亏比等）
同时将在 `plot/` 目录下生成收益曲线图，在 `state/` 目录下生成可供增量回测的状态文件（`.pkl` 和 `.json`）。