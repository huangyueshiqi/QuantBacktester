import math
import backtrader as bt

class TradeExecutor:
    """
    交易执行组件
    
    负责执行买入和卖出操作
    """
    
    def __init__(self, strategy, params):
        """
        初始化交易执行组件
        
        参数:
            strategy: 策略对象，用于访问broker和数据
            params: 策略参数
        """
        self.strategy = strategy
        self.params = params
        self.order_list = []
    
    def to_buy(self, stock_code, bs_data, ratio):
        """
        执行买入操作
        
        参数:
            stock_code: 股票代码
            bs_data: 股票数据
            ratio: 买入权重
        """
        # 是否涨停 close离涨停价小于2%认为不具备成交条件
        try:
            if bs_data.limit[1] * self.params.high_limit < bs_data.high[1]:
                print(f'{stock_code}涨停，无法买入')
                return
        except IndexError:
            # 捕捉到索引错误，避免在最后一个bar访问close[1]
            print("处理涨停股票时，Reached last bar,skipping future data access")
            return

        #计算股票数量
        raw_buy_unit=(self.strategy.broker.getvalue()* ratio * self.params.max_position)/ bs_data.close[0]

        #根据配置决定是否调整为100的整数倍
        if self.params.round_to_hundred:
            buy_unit = math.floor(raw_buy_unit/100)*100
        else:
            buy_unit = raw_buy_unit

        self.strategy.log(f"买入股票:{stock_code},权重：{ratio},金额：{self.strategy.broker.getvalue() * ratio * self.params.max_position}")

        # # 判断成交量是否满足可买入条件
        # if buy_unit > bs_data.volume[1] * 100 * self.params.vol_percent:
        #     print(f'{stock_code}成交量不足，无法买入')
        #     return
        
        # 确定执行方式
        if self.params.execution_price == 'open':
            exectype = bt.Order.Market
        elif self.params.execution_price == 'close':
            exectype = bt.Order.Close
        
        # 执行买入
        order = self.strategy.buy(data=bs_data, size=buy_unit, exectype=exectype)
        self.order_list.append(order)
        return order
    
    def to_sell(self, stock_code, bs_data, ratio):
        """
        执行卖出操作
        
        参数:
            stock_code: 股票代码
            bs_data: 股票数据
            ratio: 卖出权重
        """
        # 是否跌停 close离跌停价小于2%（low_stopping）认为不具备成交条件
        try:
            if bs_data.stopping[1] * self.params.low_stopping > bs_data.low[1]:
                print(f'{stock_code}跌停，无法卖出')
                return
        except IndexError:
            # 捕捉到索引错误，避免在最后一个bar访问close[1]
            print("处理跌停股票时，Reached last bar,skipping future data access")
            return

        # 计算股票数量
        raw_sell_unit = (self.strategy.broker.getvalue() * ratio * self.params.max_position) / bs_data.close[0]

        # 根据配置决定是否调整为100的整数倍
        if self.params.round_to_hundred:
            sell_unit = math.floor(raw_sell_unit / 100) * 100
        else:
            sell_unit = raw_sell_unit
        
        # 判断成交量是否满足可sell条件
        # if sell_unit > bs_data.volume[1] * 100 * self.params.vol_percent:
        #     print(f'{stock_code}成交量不足，无法卖出')
        #     return
        
        # 确定执行方式
        if self.params.execution_price == 'open':
            exectype = bt.Order.Market
        elif self.params.execution_price == 'close':
            exectype = bt.Order.Close
        
        # 执行卖出
        order = self.strategy.sell(data=bs_data, size=sell_unit, exectype=exectype)
        self.order_list.append(order)
        return order

    
    def sell_st_stocks(self, cur_st_list):
        """
        卖出持仓中的ST股票
        
        参数:
            cur_st_list: 当前ST股票列表
        """
        for stock in self.strategy.portfolio.get_holding_stocks():
            if stock in cur_st_list:
                try:
                    sell_data = self.strategy.getdatabyname(stock)
                    # 是否停牌
                    if sell_data.tradestatus[0] == 0:
                        print(stock + '停牌')
                        continue
                    # 跌停处理
                    if sell_data.stopping[1] * self.params.low_stopping > sell_data.low[1]:
                        print(f'st股票{stock}跌停，卖不出')
                        continue
                    
                    # 获取当前持仓数量
                    actual_position_size = self.strategy.portfolio.get_stock_total_size(stock)
                    
                    # 执行卖出
                    order = self.strategy.close(data=sell_data)
                    self.order_list.append(order)
                    print(f'st股票{stock}在{self.strategy.datas[0].datetime.date(0)}尝试卖出，当前持仓数量{actual_position_size}')
                except IndexError:
                    # 捕捉到索引错误，避免在最后一个bar访问close[1]
                    print("处理st股票时，Reached last bar,skipping future data access")
                    return
    
    def cancel_pending_orders(self):
        """
        取消所有的订单
        """
        if len(self.order_list) > 0:
            for order in self.order_list:
                self.strategy.cancel(order)  # 如果订单未完成，则撤销订单
            self.order_list = []  # 重置订单列表
    
    def rebalance_portfolio(self, cur_date, weights, dict_weights_raw):
        """
        执行投资组合再平衡
        
        参数:
            cur_date: 当前日期
            weights: 目标权重
            dict_weights_raw: 历史权重
            cur_st_list: 当前ST股票列表
        """
        # 取消未完成的订单
        self.cancel_pending_orders()
        
        # 合并股票列表，确保所有股票都处理
        all_codes = set(weights['code']).union(dict_weights_raw.keys())
        
        # 处理所有股票的买卖逻辑
        operations = []
        for stock_code in all_codes:
            bs_data = self.strategy.getdatabyname(stock_code)
            # 是否停牌
            if bs_data.tradestatus[0] == 0:
                print(f'在{cur_date}股票{stock_code}停牌')
                continue
            
            # 获取目标权重
            if stock_code in weights['code'].values:
                target_weight = weights.loc[weights['code'] == stock_code, 'weight'].iloc[0]
            else:
                target_weight = 0
            
            # 获取历史权重
            history_weight = dict_weights_raw.get(stock_code, 0)
            
            # 计算净权重差异
            weight_diff = target_weight - history_weight
            operations.append((stock_code, weight_diff, bs_data))

        def _close_all(stock_code, bs_data):
            try:
                if bs_data.stopping[1] * self.params.low_stopping > bs_data.low[1]:
                    print(f'{stock_code}跌停，无法卖出')
                    return
            except IndexError:
                print("处理跌停股票时，Reached last bar,skipping future data access")
                return
            order = self.strategy.close(data=bs_data)
            self.order_list.append(order)

        # 先执行卖出操作，释放资金，先卖大单
        sell_operations = sorted([op for op in operations if op[1] < 0], key=lambda x: (x[1], x[0]))
        for stock_code, weight_diff, bs_data in sell_operations:
            if weight_diff <= 0:
                pos_size = abs(float(self.strategy.getposition(bs_data).size))
                if pos_size <= 0:
                    continue
                target_weight = 0.0
                if stock_code in weights['code'].values:
                    target_weight = float(weights.loc[weights['code'] == stock_code, 'weight'].iloc[0])
                if target_weight <= 0:
                    _close_all(stock_code, bs_data)
                    continue

                ratio = -weight_diff
                raw_sell_unit = (self.strategy.broker.getvalue() * ratio * self.params.max_position) / bs_data.close[0]
                sell_unit = math.floor(raw_sell_unit / 100) * 100 if self.params.round_to_hundred else raw_sell_unit
                sell_unit = min(float(sell_unit), pos_size)
                if sell_unit <= 0:
                    continue
                dust_threshold = 100.0 if self.params.round_to_hundred else 0.0
                if pos_size - sell_unit < dust_threshold:
                    _close_all(stock_code, bs_data)
                    continue

                if self.params.execution_price == 'open':
                    exectype = bt.Order.Market
                elif self.params.execution_price == 'close':
                    exectype = bt.Order.Close
                else:
                    exectype = bt.Order.Market
                order = self.strategy.sell(data=bs_data, size=sell_unit, exectype=exectype)
                self.order_list.append(order)

        # 再执行买入操作，先买大单
        buy_operations = sorted([op for op in operations if op[1] > 0], key=lambda x: (x[1], x[0]), reverse=True)
        for stock_code, weight_diff, bs_data in buy_operations:
            if weight_diff > 0:
                self.to_buy(stock_code, bs_data, weight_diff)


    def topk_sell(self, stock_code, bs_data):
        """
        执行卖出操作

        参数:
            stock_code: 股票代码
            bs_data: 股票数据
        """
        # 是否跌停 close离跌停价小于2%（low_stopping）认为不具备成交条件
        try:
            if bs_data.stopping[1] * self.params.low_stopping > bs_data.low[1]:
                print(f'{stock_code}跌停，无法卖出')
                return
        except IndexError:
            # 捕捉到索引错误，避免在最后一个bar访问close[1]
            print("处理跌停股票时，Reached last bar,skipping future data access")
            return

        # 执行卖出
        order = self.strategy.close(data=bs_data)
        self.order_list.append(order)
        return order


    def adjust_portfolio_topk(self, cur_date, stocks_to_sell,stocks_to_buy,dict_weights_raw,st_stock_sell):
        """
        执行Topk策略投资组合调仓

        参数:
            cur_date: 当前日期
            stocks_to_sell: 卖出列表
            stocks_to_buy: 买入列表
        """
        # 取消未完成的订单
        self.cancel_pending_orders()

        #计算策略性卖出股票释放的总权重
        strategy_sold_weight=sum(dict_weights_raw.get(stock) for stock in stocks_to_sell)

        #计算ST股票卖出释放的权重
        st_sold_weight=sum(dict_weights_raw.get(stock) for stock in st_stock_sell)

        total_released_weight=strategy_sold_weight+st_sold_weight

        #卖出股票，全卖
        for stock_code in stocks_to_sell:
            bs_data = self.strategy.getdatabyname(stock_code)
            # 是否停牌
            if bs_data.tradestatus[0] == 0:
                print(f'在{cur_date}股票{stock_code}停牌')
                continue
            self.topk_sell(stock_code,bs_data)

        # 处理所有股票的买逻辑
        for stock_code in stocks_to_buy:
            bs_data = self.strategy.getdatabyname(stock_code)
            # 是否停牌
            if bs_data.tradestatus[0] == 0:
                print(f'在{cur_date}股票{stock_code}停牌')
                continue
            ratio=total_released_weight/len(stocks_to_buy)
            self.to_buy(stock_code,bs_data,ratio)


    def to_buy_index(self, stock_code, bs_data, ratio):
        """
        执行买入操作

        参数:
            index_code: 指数代码
            bs_data: 指数数据
            ratio: 买入权重
        """

        # 100整数倍指数数量
        buy_unit = math.floor(
            (self.strategy.broker.getvalue() * ratio * self.params.max_position) / bs_data.close[0] / 100) * 100

        # 确定执行方式
        if self.params.execution_price == 'open':
            exectype = bt.Order.Market
        elif self.params.execution_price == 'close':
            exectype = bt.Order.Close

        # 执行买入
        order = self.strategy.buy(data=bs_data, size=buy_unit, exectype=exectype)
        self.order_list.append(order)
        return order

    def to_sell_index(self, stock_code, bs_data, ratio):
        """
        执行卖出操作

        参数:
            index_code: 指数代码
            bs_data: 指数数据
            ratio: 卖出权重
        """

        # 100整数倍指数数量
        sell_unit = math.floor(
            (self.strategy.broker.getvalue() * ratio * self.params.max_position) / bs_data.close[0] / 100) * 100

        # 确定执行方式
        if self.params.execution_price == 'open':
            exectype = bt.Order.Market
        elif self.params.execution_price == 'close':
            exectype = bt.Order.Close

        # 执行卖出
        order = self.strategy.sell(data=bs_data, size=sell_unit, exectype=exectype)
        self.order_list.append(order)
        return order

    def index_rebalance_portfolio(self, cur_date, weights, dict_weights_raw):
        """
        执行指数投资组合再平衡

        参数:
            cur_date: 当前日期
            weights: 目标权重
            dict_weights_raw: 历史权重
            cur_st_list: 当前ST股票列表
        """
        # 取消未完成的订单
        self.cancel_pending_orders()

        # 合并指数列表，确保所有指数都处理
        all_indexs = set(weights['indexcode']).union(dict_weights_raw.keys())

        # 处理所有指数的买卖逻辑
        operations = []
        for index_code in all_indexs:
            bs_data = self.strategy.getdatabyname(str(index_code))

            # 获取目标权重
            if index_code in weights['indexcode'].values:
                target_weight = weights.loc[weights['indexcode'] == index_code, 'weight'].iloc[0]
            else:
                target_weight = 0

            # 获取历史权重
            history_weight = dict_weights_raw.get(index_code, 0)

            # 计算净权重差异
            weight_diff = target_weight - history_weight
            operations.append((index_code, weight_diff, bs_data))

        # 先执行卖出操作，释放资金，先卖大单
        sell_operations = sorted([op for op in operations if op[1] < 0], key=lambda x: (x[1], x[0]))
        for index_code, weight_diff, bs_data in sell_operations:
            if weight_diff < 0:
                self.to_sell_index(index_code, bs_data, -weight_diff)

        # 再执行买入操作，先买大单
        buy_operations = sorted([op for op in operations if op[1] > 0], key=lambda x: (x[1], x[0]), reverse=True)
        for index_code, weight_diff, bs_data in buy_operations:
            if weight_diff > 0:
                self.to_buy_index(index_code, bs_data, weight_diff)
