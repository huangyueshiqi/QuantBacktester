import json
import pickle
from enum import Enum


class RestorePhase(Enum):
    """恢复阶段枚举"""
    POSITIONS='positions'
    CASH_AND_PORTFOLIO='cash_and_portfolio'
    DIVIDEND_TAX='dividend_tax'
    COMPLETED='completed'

class StateManager:
    """
    状态管理组件
    
    负责管理策略状态的保存和恢复
    """
    
    def __init__(self, strategy, params):
        """
        初始化状态管理组件
        
        参数:
            strategy: 策略对象，用于访问broker和数据
            params: 策略参数
        """
        self.strategy = strategy
        self.params = params
        self.current_phase=RestorePhase.POSITIONS
        self._portfolio_state_cache=None
    
    def save_state(self, cur_date, benchmark_portfolio_daily, portfolio, dividend_tax,
                   reward, rewardcnt, risk, riskcnt):
        """
        保存策略状态
        
        参数:
            cur_date: 当前日期
            benchmark_portfolio_daily: 基准投资组合每日数据
            portfolio:Portfolio对象，包含持仓和分红记录
            dividend_tax: 分红扣税记录
            reward: 交易盈利数
            rewardcnt: 交易盈利次数
            risk: 交易亏损数
            riskcnt: 交易亏损次数
        """
        if not self.params.save_state:
            print(f'本次回测不是增量回测,不存状态')
            return
        
        # 保存绘图数据
        plot_data = benchmark_portfolio_daily[['value', 'value_daily_ratio']]
        plot_data.to_csv(f'result/plot_{cur_date}.csv', index=False)
        
        # 保存策略状态
        state = {
            'cash': self.strategy.broker.get_cash(),
            'value': self.strategy.broker.getvalue(),
        }
        
        # 保存盈亏数据
        # win_loss_dict = {
        #     'wins': rewardcnt,
        #     'losses': riskcnt,
        #     'gross_profits': reward,
        #     'gross_losses': risk,
        # }
        
        # 写入文件
        with open(f'state/state{cur_date}.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        # with open(f'state/win_loss_dict{cur_date}.pkl', 'wb') as f:
        #     pickle.dump(win_loss_dict, f)
        
        #保存Portfolio对象
        portfolio_state=portfolio.to_dict()
        with open(f"state/portfolio{cur_date}.json",'w') as f:
            json.dump(portfolio_state,f)
            print(f"Portfolio状态已保存")
        
        with open(f'state/dividend_tax{cur_date}.json', 'w') as f:
            json.dump(dividend_tax, f)
            print(f'分红扣税状态已保存')
        
        print(f"增量回测存储{cur_date}日的状态完成")

    def extract_position(self,portfolio_state):
        positions={}
        for stock_code,position_data in portfolio_state.items():
            total_size=sum(pos_data['size'] for pos_data in position_data.values())
            if total_size>0:
                positions[stock_code]=total_size
        return positions


    def restore_state(self,date,to_restore,portfolio,dividend_tax,is_first_run):
        """
        恢复策略状态

        参数：
            date:恢复状态的日期
            to_restore：要恢复的状态
            portfolio：Portfolio对象，包含持仓记录
            dividend_tax:分红扣税记录
            is_first_run：是否第一次运行

        返回：
            是否需要继续恢复状态
        """
        if is_first_run:
            return False

        #使用统一的阶段处理器
        phase_handlers={
            RestorePhase.POSITIONS:self._restore_positions,
            RestorePhase.CASH_AND_PORTFOLIO:self._restore_cash_and_portfolio,
            RestorePhase.DIVIDEND_TAX:self._restore_dividend_tax,
            RestorePhase.COMPLETED:lambda *args:False
        }

        handler=phase_handlers[self.current_phase]
        continue_restore=handler(date,to_restore,portfolio,dividend_tax)

        return continue_restore

    def _load_portfolio_state(self,date):
        """加载并缓存portfolio状态"""
        if self._portfolio_state_cache is None:
            with open(f"state/portfolio{date}.json",'r') as f:
                self._portfolio_state_cache=json.load(f)
        return self._portfolio_state_cache


    def _restore_positions(self,date,to_restore,portfolio,dividend_tax):
        """恢复持仓"""
        self.strategy.log("恢复持仓中")

        portfolio_state=self._load_portfolio_state(date)
        positions =self.extract_position(portfolio_state)

        for data_name,position_size in positions.items():
            data=next((d for d in self.strategy.datas if d._name== data_name),None)
            if data and position_size>0:
                self.strategy.buy(data=data,size=position_size)
                print(f"股票名称{data_name}，恢复仓位为{position_size}")

        self.strategy.log('持仓恢复完成')
        self.current_phase=RestorePhase.CASH_AND_PORTFOLIO
        return True

    def _restore_cash_and_portfolio(self, date, to_restore, portfolio, dividend_tax):
        """恢复现金和Portfolio状态"""
        self.strategy.broker.set_cash(to_restore['cash'])
        print(f"恢复现金：{to_restore['cash']}")

        portfolio_state = self._load_portfolio_state(date)
        portfolio.from_dict(portfolio_state)
        self.strategy.log("Portfolio状态恢复完成")

        self.current_phase = RestorePhase.DIVIDEND_TAX
        return True

    def _restore_dividend_tax(self, date, to_restore, portfolio, dividend_tax):
        """恢复分红扣税记录"""
        if not self.params.deal_dividend:
            return False

        cur_date=self.strategy.datas[0].datetime.date(0).strftime('%Y-%m-%d')

        with open(f"state/dividend_tax{date}.json",'r') as f:
            dividend_tax.update(json.load(f))

        if dividend_tax:
            tax_date,tax=next(iter(dividend_tax.items()))
            if tax_date==cur_date:
                self.strategy.broker.add_cash(-tax)
                print(f"分红扣税完成，日期{tax_date}，金额{tax}")
            else:
                dividend_tax.clear()
                print(f"不是当前日期的分红扣税记录")
        else:
            print(f"没有扣税记录")

        self.current_phase = RestorePhase.COMPLETED
        return False








