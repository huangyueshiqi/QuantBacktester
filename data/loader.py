import cx_Oracle
import os
import pandas as pd
from abc import ABC, abstractmethod
from utils.config import config
import qlib
from qlib.data import D

provider_uri = os.environ.get('QLIB_PROVIDER_URI', '/home/quant/zc/finance_deal/qlib_data/price_data0821')
qlib.init(provider_uri=provider_uri)

class BaseDataLoader(ABC):
    """
    数据加载器抽象基类
    
    定义了数据加载的通用接口，遵循单一职责原则和开闭原则
    子类可以实现不同的数据源加载而不影响外部调用
    """
    
    @abstractmethod
    def load_data(self, **kwargs):
        """
        加载数据
        
        参数:
            **kwargs: 可变参数字典
            
        返回:
            加载的数据
        """
        pass


class OracleDataLoader(BaseDataLoader):
    """
    Oracle数据库数据加载器
    
    用于从Oracle数据库加载各种股票数据
    """
    
    def __init__(self, db_type='jylh'):
        """
        初始化Oracle数据加载器
        
        参数:
            username: 数据库用户名
            password: 数据库密码
            host: 数据库主机地址
            service: 数据库服务名
        """
        if db_type == 'jylh':
            username=config.database.jylh_username
            password = config.database.jylh_password
            host = config.database.jylh_host
            service = config.database.jylh_service
        elif db_type=='wind':
            username = config.database.wind_username
            password = config.database.wind_password
            host = config.database.wind_host
            service = config.database.wind_service
        self.connection_string = f'{username}/{password}@{host}/{service}'
    
    def _execute_query(self, sql, params=None):
        """
        执行SQL查询
        
        参数:
            sql: SQL查询语句
            params: 查询参数
            
        返回:
            查询结果DataFrame
        """
        connection = cx_Oracle.connect(self.connection_string)
        cursor = connection.cursor()
        
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
                
            columns = [col[0] for col in cursor.description]
            results = cursor.fetchall()
            df = pd.DataFrame(results, columns=columns)
            
            return df
        finally:
            cursor.close()
            connection.close()
    
    def load_data(self, data_type, **kwargs):
        """
        加载指定类型的数据
        
        参数:
            data_type: 数据类型
            **kwargs: 其他参数
            
        返回:
            加载的数据DataFrame
        """
        if data_type == 'stock_detail':
            return self.get_stock_detail_data(kwargs.get('start_date'), kwargs.get('end_date'))
        elif data_type=='stock_detail_simple':
            return self.get_stock_detail_data_simple(kwargs.get('start_date'), kwargs.get('end_date'),kwargs.get('codes_list'))
        elif data_type == 'st_stock':
            return self.get_st_data(kwargs.get('start_date'), kwargs.get('end_date'))
        elif data_type == 'divident':
            return self.get_divident_data(kwargs.get('start_date'), kwargs.get('end_date'))
        elif data_type == 'proright':
            return self.get_proright_data(kwargs.get('start_date'), kwargs.get('end_date'))
        elif data_type == 'base_stock_detail':
            return self.get_base_stock_detail_data(kwargs.get('start_date'), kwargs.get('end_date'), kwargs.get('index_value'))
        elif data_type == 'benchmark_detail':
            return self.get_benchmark_detail_data(kwargs.get('start_date'), kwargs.get('end_date'), kwargs.get('index_value'))
        elif data_type == 'calendar':
            return self.get_a_calendar_data()
        elif data_type == 'index_code':
            return self.get_index_code_data(kwargs.get('start_date'), kwargs.get('end_date'))
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
    
    def get_stock_detail_data(self, start_date, end_date):
        """
        从股票行情表中获取股票行情数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            股票行情数据DataFrame
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        sql = ("select ID, TRADEDATE, STOCKCODE, OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, PCHG, VWAP, FACTOR, LCLOSE, "
               "UPLIMITPRICE, DOWNLIMITPRICE, TRADESTATUS, MARKETVALUE, TOPEN, THIGH, TLOW, TCLOSE, "
               "INDECODE, CREATETIME from JY_QY_STOCKQUOTATION_CS "
               "where TRADEDATE between :start_date and :end_date order by TRADEDATE asc")
        
        return self._execute_query(sql, (start_date, end_date))

    def get_stock_detail_data_simple(self, start_date, end_date,code_list):
        """
        从股票行情表中获取股票行情数据

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            股票行情数据DataFrame
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        placeholders = ','.join(f':c{i}' for i in range(len(code_list)))

        sql = ("select ID, TRADEDATE, STOCKCODE, OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, PCHG, VWAP, FACTOR, LCLOSE, "
               "UPLIMITPRICE, DOWNLIMITPRICE, TRADESTATUS, MARKETVALUE, TOPEN, THIGH, TLOW, TCLOSE, "
               "INDECODE, CREATETIME from JY_QY_STOCKQUOTATION_CS "
               f"where STOCKCODE IN ({placeholders}) and  TRADEDATE between :start_date and :end_date  order by TRADEDATE asc")

        params = {
            'start_date': start_date,
            'end_date': end_date
        }

        # 将代码列表填入字典：{'c0': '000001', 'c1': '000002', ...}
        for i, code in enumerate(code_list):
            params[f'c{i}'] = code
        # 4. 执行查询
        return self._execute_query(sql, params)
    
    def get_st_data(self, start_date=None, end_date=None):
        """
        获取ST股票数据
        
        参数:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        返回:
            ST股票数据DataFrame
        """
        sql = "select * from JY_QY_STDATA order by SELECTEDDATE asc"
        return self._execute_query(sql)
    
    def get_divident_data(self, start_date=None, end_date=None):
        """
        获取期间分红数据
        
        参数:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        返回:
            分红数据DataFrame
        """
        sql = ("select * from JY_QY_STOCKDIVIDENTS_CS "
               "where (GRAOBJTYPE = '1' or GRAOBJTYPE = '2') and DIVITYPE in ('1','5','15','18','58','158')")
        
        return self._execute_query(sql)
    
    def get_proright_data(self, start_date=None, end_date=None):
        """
        获取期间拆股数据
        
        参数:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        返回:
            拆股数据DataFrame
        """
        sql = "select * from JY_QY_STOCKSPLITDATA_CS where CHANGERT<=1 and LISTDATE>'19800101'"
        df = self._execute_query(sql)
        df.dropna(subset=['STOCKCODE', 'CHANGERT'], inplace=True)
        return df
    
    def get_base_stock_detail_data(self, start_date, end_date, index_value):
        """
        从基准成分股数据表中获取成分股数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            stock_index: 指数类型
            
        返回:
            基准成分股数据DataFrame
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        sql = "select * from JY_QY_BMCSTOCKS_CS where INDEXCODE=:index_value AND TRADEDATE between :start_date and :end_date"
        df = self._execute_query(sql, (index_value, start_date, end_date))
        
        # 格式化日期
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        return df[['TRADEDATE', 'STOCKCODE', 'WEIGHT', 'MARKETVALUE', 'INDEXCODE', 'INDCODE']]
    
    def get_benchmark_detail_data(self, start_date, end_date, index_value):
        """
        获取指数数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            index_value: 指数代码
            
        返回:
            指数数据DataFrame
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        sql = ("select * from JY_QY_INDEXDATA_CS "
               "where INDEXCODE=:index_value AND TRADEDATE between :start_date and :end_date order by TRADEDATE asc")
        
        df = self._execute_query(sql, (index_value, start_date, end_date))
        
        # 格式化日期
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        return df[['TRADEDATE', 'TCLOSE']]
    
    def get_a_calendar_data(self):
        """
        获取中国A股交易日历数据
        
        返回:
            交易日历数据DataFrame
        """
        connection = cx_Oracle.connect('wind', 'wind', '10.6.60.114:1521/wind')
        cursor = connection.cursor()
        
        try:
            # 读取中国A股交易日历数据
            sql = "SELECT * FROM ASHARECALENDAR order by TRADE_DAYS"
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            results = cursor.fetchall()
            df = pd.DataFrame(results, columns=columns)
            return df
        finally:
            cursor.close()
            connection.close()

    def get_index_code_data(self, start_date, end_date):
        """
        从股票行情表中获取股票行情数据

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            股票行情数据DataFrame
        """
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        sql = ("select *  from JY_QY_INDEXDATA_CS where TRADEDATE between :start_date and :end_date order by TRADEDATE asc")

        return self._execute_query(sql, (start_date, end_date))



class StockDataService:
    """
    股票数据服务
    
    整合各种数据加载器，提供统一的数据访问接口
    """
    
    def __init__(self, oracle_loader=None):
        """
        初始化股票数据服务
        
        参数:
            oracle_loader: Oracle数据加载器
        """
        self.oracle_loader = oracle_loader or OracleDataLoader()
        self.qlib_initialized=False



    def _load_stock_detail_data_by_qlib(self, start_date, end_date, codes_list=None):
        instruments = 'all'
        if codes_list:
            instruments=codes_list

        fields = [
            '$change', '$volume', '$amount', '$adjpreclose', '$open', '$high', '$low', '$close', '$factor', '$vwap',
            '$s_dq_preclose_ashareeodprices', '$s_dq_open_ashareeodprices', '$s_dq_high_ashareeodprices', '$s_dq_low_ashareeodprices', '$s_dq_close_ashareeodprices',
            '$s_dq_pctchange_ashareeodprices','$s_dq_tradestatuscode_ashareeodprices', '$s_dq_limit_ashareeodprices', '$s_dq_stopping_ashareeodprices', '$s_dq_adjclose_backward_ashareeodprices',
            '$s_val_mv_ashareeodderivativeindicator'
        ]


        stock_detail_data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_date,
            end_time=end_date,
            freq='day'
        )
        stock_detail_data = stock_detail_data.reset_index()
        stock_detail_data = stock_detail_data.rename(columns={
            'instrument': 'STOCKCODE',
            'datetime': 'TRADEDATE',
            '$open': 'OPEN',
            '$high': 'HIGH',
            '$low': 'LOW',
            '$close': 'CLOSE',
            '$volume': 'VOLUME',
            '$amount': 'AMOUNT',
            '$vwap': 'VWAP',
            '$factor': 'FACTOR',
            '$adjpreclose': 'LCLOSE',
            '$change': 'CHANGE',
            '$s_dq_limit_ashareeodprices': 'UPLIMITPRICE',
            '$s_dq_stopping_ashareeodprices': 'DOWNLIMITPRICE',
            '$s_dq_tradestatuscode_ashareeodprices': 'TRADESTATUS',
            '$s_val_mv_ashareeodderivativeindicator': 'MARKETVALUE',
            '$s_dq_open_ashareeodprices': 'TOPEN',
            '$s_dq_high_ashareeodprices': 'THIGH',
            '$s_dq_low_ashareeodprices': 'TLOW',
            '$s_dq_close_ashareeodprices': 'TCLOSE',
            '$s_dq_pctchange_ashareeodprices': 'PCHG',
            '$s_dq_preclose_ashareeodprices': 'S_DQ_PRECLOSE',
            '$s_dq_adjclose_backward_ashareeodprices': 'S_DQ_ADJCLOSE_BACKWARD'
        })

        stock_detail_data['TRADEDATE'] = pd.to_datetime(stock_detail_data['TRADEDATE']).dt.strftime('%Y-%m-%d')
        if 'TRADESTATUS' not in stock_detail_data.columns:
            stock_detail_data['TRADESTATUS'] = 1
        stock_detail_data['INDECODE'] = 0

        return stock_detail_data

    def _load_benchmark_by_source(self, start_date, end_date, index_value, stock_detail_data=None):
        benchmark_source = getattr(config.backtest.benchmark, 'source', 'index_code')
        if benchmark_source == 'self_stock' and stock_detail_data is not None:
            benchmark_detail_data = stock_detail_data[['TRADEDATE', 'TCLOSE']].copy()
            # benchmark_detail_data.rename(columns={"TOPEN":"TCLOSE"},inplace=True)
            benchmark_detail_data['TRADEDATE'] = pd.to_datetime(benchmark_detail_data['TRADEDATE']).dt.strftime(
                '%Y-%m-%d')
            benchmark_detail_data = benchmark_detail_data.sort_values('TRADEDATE').drop_duplicates(subset=['TRADEDATE'])
            return benchmark_detail_data

        if benchmark_source == 'index_code':
            index_instrument = getattr(config.backtest.benchmark, 'index_instrument', '399997.SZ')
            benchmark_detail_data = D.features(
                instruments=[index_instrument],
                fields=['$close'],
                start_time=start_date,
                end_time=end_date,
                freq='day',
            ).reset_index()
            benchmark_detail_data.rename(columns={'$close': 'TCLOSE', 'datetime': 'TRADEDATE'}, inplace=True)
            benchmark_detail_data.drop(columns=['instrument'], inplace=True)
            benchmark_detail_data['TRADEDATE']=pd.to_datetime(benchmark_detail_data['TRADEDATE'])
            benchmark_detail_data['TRADEDATE'] = benchmark_detail_data['TRADEDATE'].dt.strftime('%Y-%m-%d')
            return benchmark_detail_data

        benchmark_detail_data = self.oracle_loader.load_data(
            'benchmark_detail',
            start_date=start_date,
            end_date=end_date,
            index_value=index_value
        )
        return benchmark_detail_data


    def get_stock_data_for_backtest(self, start_date, end_date, codes_list=None, index_value='2070000191',is_finchina=False):
        """
        获取回测所需的所有股票数据

        参数:
            start_date: 开始日期
            end_date: 结束日期
            codes_list: 股票代码列表
            stock_index: 指数类型

        返回:
            回测所需的所有数据字典
        """

        # 从数据库加载数据
        # if codes_list and len(codes_list)>100:
        #     stock_detail_data = self.oracle_loader.load_data('stock_detail', start_date=start_date, end_date=end_date)
        # else:
        #     stock_detail_data = self.oracle_loader.load_data('stock_detail_simple', start_date=start_date,
        #                                                      end_date=end_date,codes_list=codes_list)

        stock_detail_data=self._load_stock_detail_data_by_qlib(start_date,end_date,codes_list)
        stock_detail_data['MARKETVALUE'].fillna(0,inplace=True)
        stock_detail_data=stock_detail_data.round(4)
        st_data = self.oracle_loader.load_data('st_stock')
        divident_data = self.oracle_loader.load_data('divident')
        proright_data = self.oracle_loader.load_data('proright')

        base_stock_detail_data = self.oracle_loader.load_data('base_stock_detail',
                                                              start_date=start_date,
                                                              end_date=end_date,
                                                              index_value=index_value)
        benchmark_detail_data = self._load_benchmark_by_source(start_date=start_date,end_date=end_date,index_value=index_value,stock_detail_data=stock_detail_data)

        # 按股票池过滤数据
        if codes_list:
            if not is_finchina:
                stock_detail_data = stock_detail_data[stock_detail_data['STOCKCODE'].isin(codes_list)]
                st_data = st_data[st_data['STOCKCODE'].isin(codes_list)]
                divident_data = divident_data[divident_data['STOCKCODE'].isin(codes_list)]
            else:
                stock_detail_data['STOCKCODE'] = stock_detail_data['STOCKCODE'].str.extract(r'(\d{6})')
                st_data['STOCKCODE'] = st_data['STOCKCODE'].str.extract(r'(\d{6})')
                divident_data['STOCKCODE'] = divident_data['STOCKCODE'].str.extract(r'(\d{6})')
                proright_data['STOCKCODE'] = proright_data['STOCKCODE'].str.extract(r'(\d{6})')
                base_stock_detail_data['STOCKCODE'] = base_stock_detail_data['STOCKCODE'].str.extract(r'(\d{6})')
                stock_detail_data = stock_detail_data[stock_detail_data['STOCKCODE'].isin(codes_list)]
                st_data = st_data[st_data['STOCKCODE'].isin(codes_list)]
                divident_data = divident_data[divident_data['STOCKCODE'].isin(codes_list)]
        else:
            stock_detail_data['STOCKCODE'] = stock_detail_data['STOCKCODE'].str.extract(r'(\d{6})')
            st_data['STOCKCODE'] = st_data['STOCKCODE'].str.extract(r'(\d{6})')
            divident_data['STOCKCODE'] = divident_data['STOCKCODE'].str.extract(r'(\d{6})')
            proright_data['STOCKCODE'] = proright_data['STOCKCODE'].str.extract(r'(\d{6})')
            base_stock_detail_data['STOCKCODE'] = base_stock_detail_data['STOCKCODE'].str.extract(r'(\d{6})')


        return {
            'stock_detail_data': stock_detail_data.round(4),
            'st_data': st_data,
            'divident_data': divident_data,
            'proright_data': proright_data,
            'base_stock_detail_data': base_stock_detail_data.round(4),
            'benchmark_detail_data': benchmark_detail_data.round(4)
        }


    def get_indexcode_data_for_backtest(self, start_date, end_date,  index_value='2070000191'):
        """
        获取回测所需的所有指数数据

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            回测所需的所有数据字典
        """

        # 从数据库加载数据
        indexcode_data = self.oracle_loader.load_data('index_code', start_date=start_date, end_date=end_date)

        benchmark_detail_data = self._load_benchmark_by_source(start_date=start_date,end_date=end_date,index_value=index_value)

        return {
            'indexcode_data': indexcode_data.round(4),
            'benchmark_detail_data': benchmark_detail_data.round(4)
        }



class DataPreprocessor:
    """
    数据预处理器
    
    对加载的原始数据进行预处理
    """
    
    @staticmethod
    def preprocess_stock_detail_data(stock_detail_data, stock_predict_data):
        """
        处理股票行情明细数据
        
        参数:
            stock_detail_data: 股票行情数据
            stock_predict_data: 股票预测数据
            
        返回:
            处理后的股票数据
        """
        # 重命名列
        stock_detail_data=stock_detail_data.rename(columns={'STOCKCODE':'code','TRADEDATE':'date'})
        
        # 转换日期格式
        stock_detail_data['date'] = pd.to_datetime(stock_detail_data['date'])
        
        # 合并预测数据和行情数据
        stock_data = pd.merge(stock_predict_data, stock_detail_data, on=['date', 'code'])
        
        # 选择所需列
        stock_data = stock_data[['code', 'date', 'TOPEN', 'THIGH', 'TLOW', 'TCLOSE', 'score', 'VOLUME', 'UPLIMITPRICE',
                               'DOWNLIMITPRICE', 'TRADESTATUS', 'INDECODE', 'MARKETVALUE']]
        
        # 重命名列
        stock_data=stock_data.rename(columns={'date': 'datetime'})
        stock_data.columns = ['code', 'datetime', 'open', 'high', 'low', 'close', 'score', 'volume', 'limit', 'stopping',
                             'tradestatus', 'indecode', 'MARKETVALUE']

        # 根据实际情况处理
        stock_data.loc[:,'tradestatus'] = stock_data['tradestatus'].astype('int16')
        stock_data.loc[:,'indecode'] = stock_data['indecode'].astype('int32')
        
        # 删除包含缺失值的行
        # stock_data.dropna(axis=0, how='any', inplace=True)
        stock_data=stock_data.dropna(axis=0, how='any')
        
        # 设置索引
        stock_data.set_index('code', inplace=True)
        stock_data=stock_data.round(4)
        
        return stock_data

    @staticmethod
    def prepare_stock_data(stock_detail_data):
        """
        处理股票行情明细数据,没有预测评分数据的情况

        参数:
            stock_detail_data: 股票行情数据

        返回:
            处理后的股票数据
        """

        # 重命名列
        column_mapping={
            'STOCKCODE':'code',
            'TRADEDATE':'datetime',
            'OPEN':'open',
            'HIGH':'high',
            'LOW':'low',
            'CLOSE':'close',
            'VOLUME':'volume',
            'UPLIMITPRICE':'limit',
            'DOWNLIMITPRICE':'stopping',
            'TRADESTATUS':'tradestatus',
            'INDECODE':'indecode',
            'MARKETVALUE':'MARKETVALUE'
        }

        #直接选择并重命名需要的列
        stock_data=stock_detail_data[list(column_mapping.keys())].rename(columns=column_mapping)

        #转换日期格式
        stock_data['datetime']=pd.to_datetime(stock_data['datetime'])

        # 删除包含缺失值的行
        stock_data.dropna(inplace=True)

        # 转换数据类型
        stock_data['tradestatus'] = stock_data['tradestatus'].astype('int16')
        stock_data['indecode'] = stock_data['indecode'].astype('int32')

        # 设置索引
        stock_data.set_index('code', inplace=True)
        stock_data = stock_data.round(4)

        return stock_data

    @staticmethod
    def prepare_indexcode_data(indexcode_data):
        """
        处理指数行情明细数据

        参数:
            indexcode_data: 指数行情数据

        返回:
            处理后的指数数据
        """

        # 重命名列
        column_mapping = {
            'INDEXCODE': 'code',
            'TRADEDATE': 'datetime',
            'TOPEN': 'open',
            'THIGH': 'high',
            'TLOW': 'low',
            'TCLOSE': 'close',
            'VOLUME': 'volume',
        }

        # 直接选择并重命名需要的列
        indexcode_data = indexcode_data[list(column_mapping.keys())].rename(columns=column_mapping)

        # 转换日期格式
        indexcode_data['datetime'] = pd.to_datetime(indexcode_data['datetime'])

        # 删除包含缺失值的行
        indexcode_data.dropna(inplace=True)


        # 设置索引
        indexcode_data.set_index('code', inplace=True)
        indexcode_data=indexcode_data.round(4)

        return indexcode_data
    
    @staticmethod
    def preprocess_st_data(st_data):
        """
        处理ST股票数据为字典
        
        参数:
            st_data: ST股票数据
            
        返回:
            处理后的ST股票字典
        """
        st_dict = {}
        accumlated_stocks = set()
        
        # 将'SELECTEDDATE'列转换为日期格式
        st_data['DATE'] = pd.to_datetime(st_data['SELECTEDDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # 按日期分组处理数据
        for dt, group in st_data.groupby('DATE'):
            # 获取该日期下的所有股票代码
            stock_codes = group['STOCKCODE'].unique()
            # 将当前日期的股票代码添加到累计的股票代码集合中
            accumlated_stocks.update(stock_codes)
            # 将累计的股票代码列表保存到字典中
            st_dict[dt] = list(accumlated_stocks)
            
        return st_dict
    
    @staticmethod
    def check_for_dividends_from_df(divident_data):
        """
        处理分红数据
        
        参数:
            divident_data: 分红数据
            
        返回:
            现金分红字典、送股分红字典
        """
        # 删除股票代码为空的行
        divident_data1 = divident_data.copy()
        divident_data1 = divident_data1.dropna(subset=['STOCKCODE'])
        
        # 格式化日期列
        divident_data1['regdate'] = pd.to_datetime(divident_data1['EQURECORDDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')  # 登记日
        divident_data1['xdrdate'] = pd.to_datetime(divident_data1['XDRDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')  # 除权除息日
        divident_data1['cashdate'] = pd.to_datetime(divident_data1['CASHDVARRENDDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')  # 分红到帐日
        divident_data1['sharedate'] = pd.to_datetime(divident_data1['SHARRDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')  # 股份到账日
        
        # 转换数值列并处理分红金额
        divident_data1['cash'] = pd.to_numeric(divident_data1['AFTTAXCASHDVCNY'], errors='coerce').fillna(0) / 10  # 每股分红金额
        divident_data1['probonus'] = pd.to_numeric(divident_data1['PROBONUSRT'], errors='coerce').fillna(0) / 10  # 每股送股数量
        
        # 过滤有效代码长度(长度为9)
        # divident_data1 = divident_data1[divident_data1['STOCKCODE'].str.len() == 9]
        
        # 按股票代码和登记日期分组,并对cash和probonus字段求和
        divident_data2 = divident_data1.groupby(['STOCKCODE', 'regdate'], as_index=False).agg({
            'xdrdate': 'first',  # 保留第一行的除权除息日
            'cashdate': 'first',
            'cash': 'sum',
            'probonus': 'sum',
            'sharedate': 'first',
        })
        
        # 处理现金分红数据
        valid_data1 = divident_data2[divident_data2['cash'] > 0]
        dividends = {}
        grouped = valid_data1.groupby('STOCKCODE')
        for stock, group in grouped:
            if stock not in dividends:
                dividends[stock] = {}
            # 使用字典构造方法生成每只股票的对应数据结构
            dividends[stock].update(
                group.set_index('regdate')[['xdrdate', 'cashdate', 'cash']].assign(status='registration').to_dict('index')
            )
        
        # 处理送股分红数据
        valid_data2 = divident_data2[divident_data2['probonus'] > 0]
        dividends_probonus = {}
        grouped1 = valid_data2.groupby('STOCKCODE')
        for stock, group in grouped1:
            if stock not in dividends_probonus:
                dividends_probonus[stock] = {}
            # 使用字典构造方法生成每只股票的对应数据结构
            dividends_probonus[stock].update(
                group.set_index('regdate')[['sharedate', 'probonus']].assign(status='registration').to_dict('index')
            )
            
        return dividends, dividends_probonus
    
    @staticmethod
    def load_proright_from_df(proright_data):
        """
        处理拆股数据
        
        参数:
            proright_data: 拆股数据
            
        返回:
            拆股数据字典
        """
        # 过滤出有效的拆合缩比例数据并格式化日期
        proright_data = proright_data.copy()
        proright_data['ratio'] = pd.to_numeric(proright_data['CHANGERT'], errors='coerce').fillna(0)  # 拆合缩比例(1：X)
        proright_data['regdate'] = pd.to_datetime(proright_data['SHCAPBASEDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')  # 股本基准日
        proright_data['listdate'] = pd.to_datetime(proright_data['LISTDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')  # 上市日
        
        # 过滤有效数据(拆合缩比例>0)
        valid_data = proright_data[proright_data['ratio'] > 0]
        
        # 将数据按照股票代码分组后直接构建数据结构
        dividends_changert = {}
        grouped = valid_data.groupby('STOCKCODE')
        for stock, group in grouped:
            if stock not in dividends_changert:
                dividends_changert[stock] = {}
            # 使用字典构造方法生成每只股票的对应数据结构
            dividends_changert[stock].update(
                group.set_index('regdate')[['ratio', 'listdate']].assign(status='registration').to_dict('index')
            )
            
        return dividends_changert 