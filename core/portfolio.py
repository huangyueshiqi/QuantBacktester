import cvxpy as cp
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod


class PortfolioOptimizer(ABC):
    """
    投资组合优化器抽象基类
    
    定义了投资组合优化的通用接口，遵循单一职责原则和开闭原则
    子类可以实现不同的优化算法而不影响外部调用
    """
    
    @abstractmethod
    def optimize(self, section_data, hold_data, benchmark_data):
        """
        优化投资组合权重
        
        参数:
            section_data: 股票池数据
            hold_data: 持仓数据
            benchmark_data: 基准数据
            
        返回:
            优化后的权重DataFrame
        """
        pass
    
    @abstractmethod
    def set_params(self, **kwargs):
        """
        设置优化参数
        
        参数:
            **kwargs: 可变参数字典
        """
        pass


class CvxPortfolioOptimizer(PortfolioOptimizer):
    """
    使用CVXPY实现的投资组合优化器
    """
    
    def __init__(self, weights_upper=1, weights_offset=1, marketcaps_offset=1, 
                 industry_offset=1, turnover_upper=1,min_hs300_weight=0.6,
                 min_hs300_count_ratio=0.5,marketcap_threshold=50,small_cap_max_count=10):
        """
        初始化CVXPY优化器
        
        参数:
            weights_upper: 每只股票的权重上限[0,1]
            weights_offset: 每只股票的权重与基准的偏离[0,1%]
            marketcaps_offset: 组合与基准的市值偏离[0,10%]
            industry_offset: 组合与基准的行业偏离[0,1%]
            turnover_upper: 组合与当前持仓的换手率约束[0,1]
            min_hs300_weight:组合中沪深300成分股的权重约束[0,1]
            min_hs300_count_ratio:组合中沪深300成分股数量所占比例的约束[0,1]
            marketcap_threshold:市值阈值(亿元)，低于此市值的股票被视为小市值股票
            small_cap_max_count:小市值股票的最大持有数量，None表示不限制
        """
        self.set_params(
            weights_upper=weights_upper,
            weights_offset=weights_offset,
            marketcaps_offset=marketcaps_offset,
            industry_offset=industry_offset,
            turnover_upper=turnover_upper,
            min_hs300_weight=min_hs300_weight,
            min_hs300_count_ratio=min_hs300_count_ratio,
            marketcap_threshold=marketcap_threshold,
            small_cap_max_count=small_cap_max_count
        )
        self.weight_cnt = 0  # 优化算法运行次数计数器
        
    def set_params(self, **kwargs):
        """设置优化器参数"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def convert_df_to_np(self, section_data, hold_data, benchmark_data):
        """
        将输入的pandas数据转换成numpy数据
        
        参数:
            section_data: 股票池数据
            hold_data: 持仓数据
            benchmark_data: 基准数据
        """
        # 将持仓信息、benchmark与股票池对齐，根据section_data.index对齐
        hold_data_reindexed = hold_data.reindex(section_data.index, fill_value=0)
        benchmark_data_reindexed = benchmark_data.reindex(section_data.index, fill_value=0)

        self.expected_returns = section_data['score'].to_numpy()
        self.marketcaps = section_data['marketcaps'].to_numpy() / (10 ** 8)  # 市值单位转换成亿
        self.industry = section_data['industry'].to_numpy()
        
        self.current_weights = hold_data_reindexed['weight'].to_numpy()  # 把权重转换成0~1
        self.benchmark_weights = benchmark_data_reindexed['weight'].to_numpy()  # 把权重转换成0~1

        self.benchmark_weights_raw = benchmark_data['weight'].to_numpy()  # 把权重转换成0~1
        self.benchmark_marketcaps = benchmark_data['marketcaps'].to_numpy() / (10 ** 8)  # 市值单位转换成亿
        self.benchmark_industry = benchmark_data['industry'].to_numpy()

        self.n_stocks_pool = section_data.shape[0]  # 股票池中股票数量
        self.all_industries = section_data['industry'].unique()  # 股票池中行业代码

        self.in_hs300 = section_data['in_hs300'].to_numpy()
    
    def calculate_weights(self):
        """
        计算优化权重
        
        返回:
            numpy数组形式的权重
        """
        # 初始化变量
        weights = cp.Variable(self.n_stocks_pool)
        #二元变量，表示是否持仓该股票
        holdings=cp.Variable(self.n_stocks_pool,boolean=True)
        # 目标函数：最大化预期收益率
        objective = cp.Maximize(self.expected_returns @ weights)
        
        # 约束条件
        constraints = []
        
        # 1. 个股权重约束
        constraints.append(cp.sum(weights) == 1)
        constraints.append(weights <= self.weights_upper)
        constraints.append(weights >= 0)

        #耦合 weights和holdings
        constraints.append(weights<=self.weights_upper*holdings)
        constraints.append(weights>=0.0001*holdings)
        
        # 基准权重约束
        if self.weights_offset < 1 and self.weights_offset >= 0:
            constraints.append(weights >= self.benchmark_weights - self.weights_offset)
            constraints.append(weights <= self.benchmark_weights + self.weights_offset)
        
        # 2. 组合风格暴露约束
        if self.marketcaps_offset < 1 and self.marketcaps_offset >= 0:
            # 计算市值权重的总和
            marketcap_weight = cp.sum(weights @ self.marketcaps)

            # 计算基准市值权重的平均值
            benchmark_marketcap_mean = np.mean(self.benchmark_marketcaps)
            # 添加约束
            constraints.append(cp.abs(marketcap_weight - benchmark_marketcap_mean) <= self.marketcaps_offset * benchmark_marketcap_mean)
        
        #3. 行业约束
        # if self.industry_offset < 1 and self.industry_offset >= 0:
        #     for industry in self.all_industries:
        #         # 当前w的行业分布
        #         industry_stocks = np.where(self.industry == industry)[0]
        #         industry_weight = cp.sum(weights[industry_stocks])
        #         # 基准的行业分布
        #         industry_stocks_benchmark = np.where(self.benchmark_industry == industry)[0]
        #         benchmark_industry_weight = np.sum(self.benchmark_weights_raw[industry_stocks_benchmark])
        #         constraints.append(cp.abs(industry_weight - benchmark_industry_weight) <= self.industry_offset)
        #
        # # 4. 换手率约束
        # if self.turnover_upper < 1 and self.turnover_upper >= 0:
        #     constraints.append(cp.sum(cp.abs(weights - self.current_weights)) <= self.turnover_upper)

        # # 5. 沪深300成分股权重不低于60%
        # if self.min_hs300_weight < 1 and self.min_hs300_weight >=0:
        #     constraints.append(weights@self.in_hs300>=self.min_hs300_weight)
        #
        # # 6. 沪深300成分股数量占比约束
        # if self.min_hs300_count_ratio < 1 and self.min_hs300_count_ratio>=0:
        #     #沪深300成分股的持仓指示变量之和
        #     hs300_held_count=cp.sum(holdings @ self.in_hs300)
        #     #总持仓股票数量
        #     total_held_count=cp.sum(holdings)
        #     constraints.append(hs300_held_count>=self.min_hs300_count_ratio*total_held_count)
        #
        # # 7. 小市值股票数量约束
        # if self.marketcap_threshold>0 and self.small_cap_max_count>=0:
        #     is_small=self.marketcaps<self.marketcap_threshold
        #     small_held_count=cp.sum(holdings[is_small])
        #     constraints.append(small_held_count<=self.small_cap_max_count)
        #
        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB)   #cp.SCIP
        
        # 优化算法运行次数计算
        self.weight_cnt += 1
        print(f'优化次数{self.weight_cnt}')
        
        if problem.status != cp.OPTIMAL:
            return None
        
        weights_res = problem.variables()[0].value
        weights_res[weights_res < 0.0001] = 0


        
        return weights_res
    
    def optimize(self, section_data, hold_data, benchmark_data):
        """
        优化投资组合权重，包含自动参数调整逻辑
        
        参数:
            section_data: 股票池数据
            hold_data: 持仓数据
            benchmark_data: 基准数据
            
        返回:
            优化后的权重DataFrame
        """
        section_data['in_hs300']=section_data.index.isin(benchmark_data.index).astype(int)
        print(f"section:{section_data}")
        self.convert_df_to_np(section_data, hold_data, benchmark_data)

        weights_res = self.calculate_weights()
        
        # 若未找到最优解，进行网格搜索放松约束条件
        if weights_res is None:
            print(f"{self.weights_offset}, {self.marketcaps_offset}, {self.industry_offset}, 阈值无解，开始搜索...")
            # 对于无解的情况，通过网格搜索进行约束放松，尝试求解
            weights_offset_tolerance = np.arange(self.weights_offset, self.weights_offset + 0.05, 0.01)  # 权重偏离容忍度
            marketcaps_offset_tolerance = np.arange(self.marketcaps_offset, self.marketcaps_offset + 0.5, 0.1)   # 市值偏离容忍度
            industry_offset_tolerance = np.arange(self.industry_offset, self.industry_offset + 0.05, 0.01)  # 行业偏离容忍度
            
            for i in industry_offset_tolerance:
                for m in marketcaps_offset_tolerance:
                    for w in weights_offset_tolerance:
                        self.set_params(weights_offset=w, marketcaps_offset=m, industry_offset=i)
                        weights_res = self.calculate_weights()
                        if weights_res is not None:
                            print(f"{w}, {m}, {i}, 调整阈值找到解")
                            break
                    if weights_res is not None:
                        break
                if weights_res is not None:
                    break
                    
            if weights_res is None:
                print("无法找到解决方案！")
                return None

        
        # 转换结果为DataFrame
        result_df=pd.DataFrame({'code':section_data.index,'weight':weights_res}).query('weight>0')
        result_df=result_df.reset_index(drop=True)
        
        print(f"总权重: {result_df['weight'].sum()}")
        print(result_df)

        # result_df['in_hs300']=result_df['code'].isin(benchmark_data.index).astype(int)
        # hs300_weight_sum=(result_df['weight']*result_df['in_hs300']).sum()
        # n_hs300=sum(result_df['in_hs300'])
        # n_total=len(result_df)
        # print(f"沪深300成分股权重合计：{hs300_weight_sum:.4f}")
        # print(f"总数：{n_total}，沪深300成分股数量：{n_hs300}，bili:{n_hs300/n_total}")
        # section_data=section_data.reset_index()
        # section_data=section_data.rename(columns={'index':'code'})
        # print(f"section_data1:{section_data}")
        # result_df=result_df.merge(section_data[['code','marketcaps']],on='code')
        # small_stocks=result_df[result_df['marketcaps']<=5000000000]
        # print(f"小市值股票：{len(small_stocks)}:{small_stocks}")
        
        return result_df


class EqualWeightOptimizer(PortfolioOptimizer):
    """
    等权重投资组合优化器
    """
    
    def __init__(self, top_k=50):
        """
        初始化等权重优化器
        
        参数:
            top_k: 选择前K只股票
        """
        self.top_k = top_k
    
    def set_params(self, **kwargs):
        """设置优化器参数"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def optimize(self, section_data, hold_data, benchmark_data):
        """
        等权重投资组合优化
        
        参数:
            section_data: 股票池数据
            hold_data: 持仓数据
            benchmark_data: 基准数据
            
        返回:
            等权重的DataFrame
        """
        # 按score排序选择前top_k只股票
        sorted_data = section_data.sort_values('score', ascending=False)
        top_stocks = sorted_data.iloc[:self.top_k]
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'code': top_stocks.index,
            'weight': 1.0 / self.top_k
        })
        
        return result


class TopkConstraintOptimizer(PortfolioOptimizer):
    """
    带指数成分股约束的投资组合优化器
    使用CVXPY实现，支持
    1.指数成分股个数、权重约束
    2.行业分布约束，确保投资组合在各个行业内合理分布
    3.市值约束，限制小市值股票的持有数量
    """

    def __init__(self, weights_upper=1, industry_offset=1,  index_constituent_stocks=None,
                 index_min_stocks=0, index_min_weight=0,marketcap_threshold=None,small_cap_max_count=None):
        """
        初始化优化器

        参数:
            weights_upper: 每只股票的权重上限[0,1]
            industry_offset: 组合与基准的行业偏离[0,1]
            index_constituent_stocks: 指数成分股列表
            index_min_stocks: 指数成分股最小持有个数
            index_min_weight: 指数成分股最小权重占比[0,1]
            marketcap_threshold:市值阈值(亿元)，低于此市值的股票被视为小市值股票
            small_cap_max_count:小市值股票的最大持有数量，None表示不限制

        """
        self.set_params(
            weights_upper=weights_upper,
            industry_offset=industry_offset,
            index_constituent_stocks=index_constituent_stocks or [],
            index_min_stocks=index_min_stocks,
            index_min_weight=index_min_weight,
            marketcap_threshold=marketcap_threshold,
            small_cap_max_count=small_cap_max_count

        )
        self.weight_cnt = 0  # 优化算法运行次数计数器

    def set_params(self, **kwargs):
        """设置优化器参数"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def convert_df_to_np(self, section_data, hold_data, benchmark_data):
        """
        将输入的pandas数据转换成numpy数据，并处理指数成分股信息

        参数:
            section_data: 股票池数据
            hold_data: 持仓数据
            benchmark_data: 基准数据
        """
        # 将持仓信息、benchmark与股票池对齐，根据section_data.index对齐
        hold_data_reindexed = hold_data.reindex(section_data.index, fill_value=0)
        benchmark_data_reindexed = benchmark_data.reindex(section_data.index, fill_value=0)

        self.expected_returns = section_data['score'].to_numpy()
        self.marketcaps = section_data['marketcaps'].to_numpy() / (10 ** 8)  # 市值单位转换成亿
        self.industry = section_data['industry'].to_numpy()
        self.stock_codes = section_data.index.to_numpy()  # 股票代码数组

        self.current_weights = hold_data_reindexed['weight'].to_numpy()  # 把权重转换成0~1
        self.benchmark_weights = benchmark_data_reindexed['weight'].to_numpy()  # 把权重转换成0~1

        self.benchmark_weights_raw = benchmark_data['weight'].to_numpy()  # 把权重转换成0~1
        self.benchmark_marketcaps = benchmark_data['marketcaps'].to_numpy() / (10 ** 8)  # 市值单位转换成亿
        self.benchmark_industry = benchmark_data['industry'].to_numpy()

        self.n_stocks_pool = section_data.shape[0]  # 股票池中股票数量
        self.all_industries = section_data['industry'].unique()  # 股票池中行业代码

        # 处理指数成分股信息
        self.index_mask = np.isin(self.stock_codes, self.index_constituent_stocks)
        self.index_stocks_count = np.sum(self.index_mask)

        print(f"股票池中指数成分股数量: {self.index_stocks_count}")
        print(f"指数成分股代码: {self.stock_codes[self.index_mask]}")

    def calculate_weights(self):
        """
        计算优化权重，包含指数成分股约束、行业约束和市值约束

        返回:
            numpy数组形式的权重
        """
        # 初始化变量
        weights = cp.Variable(self.n_stocks_pool)
        # 目标函数：最大化预期收益率
        objective = cp.Maximize(self.expected_returns @ weights)

        # 约束条件
        constraints = []

        # 1. 基本权重约束
        constraints.append(cp.sum(weights) == 1)
        constraints.append(weights <= self.weights_upper)
        constraints.append(weights >= 0)

        # 2. 行业约束
        if self.industry_offset < 1 and self.industry_offset >= 0:
            for industry in self.all_industries:
                # 当前组合的行业分布
                industry_stocks = np.where(self.industry == industry)[0]
                industry_weight = cp.sum(weights[industry_stocks])
                # 基准的行业分布
                industry_stocks_benchmark = np.where(self.benchmark_industry == industry)[0]
                benchmark_industry_weight = np.sum(self.benchmark_weights_raw[industry_stocks_benchmark])
                constraints.append(cp.abs(industry_weight - benchmark_industry_weight) <= self.industry_offset)
            print(f"添加行业约束：行业偏离上限{self.industry_offset:.2%}")

        # 3. 指数成分股约束
        if len(self.index_constituent_stocks) > 0:
            # 3.1 指数成分股个数约束
            if self.index_min_stocks > 0 and self.index_stocks_count >= self.index_min_stocks:
                # 使用二进制变量表示是否持有某只股票
                binary_vars = cp.Variable(self.n_stocks_pool, boolean=True)
                # 权重与二进制变量的关系：如果权重>0，则二进制变量=1
                constraints.append(weights <= binary_vars * self.weights_upper)
                # 指数成分股最小持有个数约束
                constraints.append(cp.sum(binary_vars[self.index_mask]) >= self.index_min_stocks)
                print(f"添加指数成分股个数约束: 最少持有 {self.index_min_stocks} 只")

            # 3.2 指数成分股权重约束
            if self.index_min_weight > 0:
                index_weight_sum = cp.sum(weights[self.index_mask])
                constraints.append(index_weight_sum >= self.index_min_weight)
                print(f"添加指数成分股权重约束: 最少权重 {self.index_min_weight:.2%}")

        #4. 市值约束-限制小市值股票数量
        if self.marketcap_threshold is not None and self.small_cap_max_count is not None:
            #识别小市值股票
            small_cap_mask=self.marketcaps<self.marketcap_threshold
            small_cap_count=np.sum(small_cap_mask)

            if small_cap_count>0:
                #小市值股票数量约束
                constraints.append(cp.sum(binary_vars[small_cap_mask])<=self.small_cap_max_count)
                print(f"添加市值约束：市值低于{self.marketcap_threshold}亿的股票最多持有{self.small_cap_max_count}只")



        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB)

        # 优化算法运行次数计算
        self.weight_cnt += 1
        print(f'优化次数{self.weight_cnt}')

        if problem.status != cp.OPTIMAL:
            print(f"优化状态: {problem.status}")
            return None

        weights_res = problem.variables()[0].value
        weights_res[weights_res < 0.0001] = 0

        # 输出指数约束满足情况
        if len(self.index_constituent_stocks) > 0:
            index_holdings = np.sum(weights_res[self.index_mask] > 0)
            index_weight = np.sum(weights_res[self.index_mask])
            print(f"\n指数成分股持有个数: {index_holdings}")
            print(f"指数成分股权重占比: {index_weight:.2%}")

        return weights_res

    def optimize(self, section_data, hold_data, benchmark_data):
        """
        优化投资组合权重，包含自动参数调整逻辑和指数约束

        参数:
            section_data: 股票池数据
            hold_data: 持仓数据
            benchmark_data: 基准数据

        返回:
            优化后的权重DataFrame
        """
        self.convert_df_to_np(section_data, hold_data, benchmark_data)

        weights_res = self.calculate_weights()

        # 若未找到最优解，进行网格搜索放松约束条件
        if weights_res is None:
            print(f"初始参数无解，开始放松约束条件...")
            # 对于无解的情况，通过网格搜索进行指数约束放松，尝试求解
            original_index_min_stocks = self.index_min_stocks
            original_index_min_weight = self.index_min_weight

            # 尝试放松指数约束
            for index_stocks_reduction in [0, 1, 2, 3, 5]:
                for index_weight_reduction in [0, 0.01, 0.02, 0.05]:
                    self.set_params(
                        index_min_stocks=max(0, original_index_min_stocks - index_stocks_reduction),
                        index_min_weight=max(0, original_index_min_weight - index_weight_reduction),
                    )
                    weights_res = self.calculate_weights()
                    if weights_res is not None:
                        print(
                            f"找到解: 指数约束: min_stocks={self.index_min_stocks}, min_weight={self.index_min_weight}")
                        break
                if weights_res is not None:
                    break

            if weights_res is None:
                print("无法找到满足指数约束的解决方案！")
                return None

        # 转换结果为DataFrame
        non_zero_indices = np.where(weights_res != 0)[0]
        codes_with_non_zero_values = section_data.index[non_zero_indices]
        non_zero_values = weights_res[non_zero_indices]

        result_df = pd.DataFrame({'code': codes_with_non_zero_values, 'weight': non_zero_values})

        print(f"总权重: {result_df['weight'].sum()}")
        print(f"选中股票数量: {len(result_df)}")

        return result_df