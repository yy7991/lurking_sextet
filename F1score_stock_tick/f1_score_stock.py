import numpy as np
import pandas as pd
from typing import Union, List, Tuple


class TimeZone_F1:
    def __init__(self, 
                factor_value:pd.Series, 
                return_value:pd.Series, 
                buy_hold_upper_limit:int = 3, 
                buy_hold_lower_limit:int = -1,
                sell_hold_upper_limit:int = -3,
                sell_hold_lower_limit:int = 1
            ):
        '''
            factor_value : 计算好的因子值Sereis。
            return_value : 计算好的收益值Sereis。

            buy_hold_upper_limit : 根据策略而变的因子/收益信号阈值相关买入-平仓区间的n*std上限。
            buy_hold_lower_limit : 根据策略而变的因子/收益信号阈值相关买入-平仓区间的n*std下限。

            sell_hold_upper_limit : 根据策略而变的因子/收益信号阈值相关卖出-平仓区间的n*std下限。
            sell_hold_lower_limit : 根据策略而变的因子/收益信号阈值相关卖出-平仓区间的n*std上限。

        '''
        self.factor_value = factor_value
        self.return_value = return_value

        self.factor_buy_hold_interval = (
            factor_value.mean() + buy_hold_upper_limit * factor_value.std(), 
            factor_value.mean() + buy_hold_lower_limit * factor_value.std()
            )
        self.return_buy_hold_interval = (
            return_value.mean() + buy_hold_upper_limit * return_value.std(), 
            return_value.mean() + buy_hold_lower_limit * return_value.std()
            )
        self.factor_sell_hold_interval = (
            factor_value.mean() + sell_hold_upper_limit * factor_value.std(), 
            factor_value.mean() + sell_hold_lower_limit * factor_value.std()
            )
        self.return_sell_hold_interval = (
            return_value.mean() + sell_hold_upper_limit * return_value.std(), 
            return_value.mean() + sell_hold_lower_limit * return_value.std()
            )
        
        self.factor_buy_hold_actions = []
        self.return_buy_hold_actions = []

        self.factor_sell_hold_actions = []
        self.return_sell_hold_actions = []

        self._record()

    def _get_TP_overlaps(self, f_index_value_list:list, r_index_value_list:list):
        '''
            用于计算正确预测的信号区间(即信号买入再平仓、卖出再平仓的区间)和收益率的同操作区间的时间累计和。理论上args1, args2参数顺序可以调换。
            f_index_value_list : 因子值的buy_hold_actions列表(记录的是索引元组)。
            r_index_value_list : 收益率的buy_hold_actions列表(记录的是索引元组)。

        '''
        overlaps = []

        # 遍历两个列表
        for start1, end1 in f_index_value_list:
            for start2, end2 in r_index_value_list:
                # 检查是否有重叠
                if start1 <= end2 and start2 <= end1:
                    # 计算重叠部分
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    overlaps.append((overlap_start, overlap_end))
        # print(overlaps)
        
        return sum([(inter_1 - inter_0) for inter_0, inter_1 in overlaps]), overlaps
    
    def _get_FX_overlaps(self, index_value_list1:list, index_value_list2:list):
        '''
            注意到对于信号时长的F1 score的计算, FN和FP分别对应收益率有的信号区间但因子值没的 和 收益率没有的区间但因子值有的。
            所以计算时只要把参数换一下位置就可以求出FN和FP对应的累计时长和区间了。
            此函数返回的是list1有的但list2没的区间。
        '''
        result = []
        for start1, end1 in index_value_list1:
            # 存储当前区间与 list2 的重叠部分
            overlaps = []
            
            # 遍历 list2，找到与当前区间的重叠部分
            for start2, end2 in index_value_list2:
                if start1 <= end2 and start2 <= end1:
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    overlaps.append((overlap_start, overlap_end))
            
            # 对重叠部分按起始时间排序
            overlaps.sort()
            
            # 找到当前区间中不在 list2 的部分
            current_start, current_end = start1, end1
            non_overlaps = []
            
            for overlap_start, overlap_end in overlaps:
                if overlap_start > current_start:
                    # 添加不重叠的部分
                    non_overlaps.append((current_start, overlap_start))
                # 更新 current_start
                current_start = max(current_start, overlap_end)
            
            # 添加最后剩余的部分
            if current_start < current_end:
                non_overlaps.append((current_start, current_end))
            
            # 将当前区间的不重叠部分添加到最终结果中
            result.extend(non_overlaps)
        return sum([(inter_1 - inter_0) for inter_0, inter_1 in result]), result

    def calculate_TP(self):
        '''
            详见 self._get_TP_overlaps
        '''
        sum_buy_TP, _ = self._get_TP_overlaps(self.factor_buy_hold_actions, self.return_buy_hold_actions)
        sum_sell_TP, _ = self._get_TP_overlaps(self.factor_sell_hold_actions, self.return_sell_hold_actions)
        print('sum_buy_tp:', sum_buy_TP)
        print('sum_sell_tp:', sum_sell_TP)
        return sum_buy_TP + sum_sell_TP
    
    def calculate_FP(self):
        '''
            详见 self._get_FX_overlaps
        '''
        sum_buy_FP, _ = self._get_FX_overlaps(self.factor_buy_hold_actions, self.return_buy_hold_actions)
        sum_sell_FP, _ = self._get_FX_overlaps(self.factor_sell_hold_actions, self.return_sell_hold_actions)
        print('sum_buy_fp:', sum_buy_FP)
        print('sum_sell_fp:', sum_sell_FP)
        return sum_buy_FP + sum_sell_FP
    
    def calculate_FN(self):
        '''
            详见 self._get_FX_overlaps
        '''
        sum_buy_FN, _ = self._get_FX_overlaps(self.return_buy_hold_actions, self.factor_buy_hold_actions)
        sum_sell_FN, _ = self._get_FX_overlaps(self.return_sell_hold_actions, self.factor_sell_hold_actions)
        print('sum_buy_fn:', sum_buy_FN)
        print('sum_sell_fn:', sum_sell_FN)
        return sum_buy_FN + sum_sell_FN
    
    def calculate_F1(self):
        tp = self.calculate_TP()
        fp = self.calculate_FP()
        fn = self.calculate_FN()
        # precision = self.calculate_TP() / (self.calculate_TP() + self.calculate_FP())
        precision = tp / (tp + fp)
        print(precision)
        # recall = self.calculate_TP() / (self.calculate_TP() + self.calculate_FN())
        recall = tp / (tp + fn)
        print(recall)
        if (precision + recall) == 0:
            raise ValueError('TP值加FP值为0')
        f1 = 2 * ((precision * recall) / (precision + recall))
        return f1

    def _record(self):
        '''
            利用递归树生成信号区间的索引, 所以要求传入的是带索引的Sereis。
        '''
        next_index(0, self.factor_value, self.factor_buy_hold_interval, self.factor_buy_hold_actions)
        next_index(0, self.factor_value, self.factor_sell_hold_interval, self.factor_sell_hold_actions)
        next_index(0, self.return_value, self.return_buy_hold_interval, self.return_buy_hold_actions)
        next_index(0, self.return_value, self.return_sell_hold_interval, self.return_sell_hold_actions)



def next_temp(index, seq, interval, index_value_list):
    '''
        递归树的子函数, 用来判断索引对应的值是否达到信号要求
    '''
    if interval[0] > interval[1]:
        try:
            buy_hold_1_index = int(seq[index:][seq[index:]>=interval[0]].index.min())
        except:
            return (index_value_list,)
        try:
            buy_hold_2_index = int(seq[buy_hold_1_index:][seq[buy_hold_1_index:]<=interval[1]].index.min())
        except:
            return (index_value_list,)
        index_value_list.append((buy_hold_1_index, buy_hold_2_index))
        new_index = buy_hold_2_index + 1
    else:
        try:
            sell_hold_1_index = int(seq[index:][seq[index:]<=interval[0]].index.min())
        except:
            return (index_value_list,)
        try:
            sell_hold_2_index = int(seq[sell_hold_1_index:][seq[sell_hold_1_index:]>=interval[1]].index.min())
        except:
            return (index_value_list,)
        index_value_list.append((sell_hold_1_index, sell_hold_2_index))
        new_index = sell_hold_2_index + 1

    return (new_index, index_value_list)


def next_index(index:int, seq:pd.Series, interval:Union[tuple, list], index_value_list:list):
    '''
        递归树本数, 用来找到一个Series的符合interval的区间。 
        index : 一般在调用的时候初始设置为0, 一般无需改动。
        seq : 传入的收益率或者因子值的Series, 索引要确保从0开始。
        index_value_list : 一般需要初始化一个空列表传入, 一般无需改动。(至于为什么不直接空置空列表默认因为python的bug不允许空列表作为默认参数。需要外部传入是因为方便类方法修改列表值的可变私有属性。)
    '''
    if not isinstance(seq, pd.Series):
        raise ValueError('seq must be a pd.Series.')
    if not isinstance(index, int):
        raise ValueError('index must be a int.')
    if interval is None or (not ((not isinstance(interval, tuple)) or (not isinstance(interval, list)))) or (len(interval) != 2):
        raise ValueError('interval must be a tuple or a list with a length of 2.')
    assert (index >= 0 and index <= len(seq)), 'index must be in the range of sequence length.'

    next_result = next_temp(index, seq, interval, index_value_list)
    if len(next_result) > 1:
        # print(next_result)
        new_index, index_value_list = next_result
    else:
        return next_result[0]
    
    if new_index == (len(seq) - 1):
        return index_value_list
    else:  
        return next_index(new_index, seq, interval, index_value_list)
    

if __name__ == '__main__':

    
    api_user_name = 'pengyuan.ding'
    api_password  = 'Ding20010111'
    from causis_api.const import get_version, login
    login.username = api_user_name
    login.password = api_password
    login.version = get_version()
    from causis_api.data import *
    from causis_api.data import *
    from causis_api.factor import *


    # 这是一个随便写的测试的因子计算代码。
    def factor_test(df_tick:pd.DataFrame, TI:int=30):  
        def weighted_sum(window):
            # 创建权重序列，从1递增到1/T
            weights = np.linspace(1, TI, TI)
            # 计算加权和
            return np.nansum(window * weights)/(weights.sum())
        ask_quant = df_tick[['ask_volume'+str(i) for i in range(1,9)]].mean(axis=1).fillna(0).rolling(TI).apply(weighted_sum, raw=True)
        # print(ask_quant.describe())
        bid_quant = df_tick[['bid_volume'+str(i) for i in range(1,9)]].mean(axis=1).fillna(0).rolling(TI).apply(weighted_sum, raw=True)
        # print(bid_quant.describe())
        f = (ask_quant-bid_quant)/(ask_quant+bid_quant+1e-5)
        f = (f - f.mean())/(f.std() + 1e-5)
        return f.rolling(TI).mean().astype(float)
    
    # 随便取一只股票数据。
    df_am = fetch_tick_data('002050', '2024-12-02','1sTick')
    #计算因子数据Series
    fv = factor_test(df_am)

    cur = df_am['current']
    # 计算收益率Sereis
    rv = ((cur.shift(-300)-cur)/(cur)).rename('rv')
    mask = cur < 1e-4
    rv = rv.mask(mask, 0)


    # 画图大致可视化一下。
    import matplotlib.pyplot as plt
    plt.plot(fv/100, label = '因子值')
    plt.plot(rv, label = '收益率')
    plt.legend()
    plt.show()

    fv_m, rv_m, fv_std, rv_std = fv.mean(), rv.mean(), fv.std(), rv.std()

    factor_buy_hold_interval = (fv_m + 2*fv_std, fv_m - 1*fv_std)
    factor_sell_hold_interval = (fv_m - 2*fv_std, fv_m + 1*fv_std)
    return_buy_hold_interval = (rv_m + 2*rv_std, rv_m - 1*rv_std)
    return_sell_hold_interval = (rv_m - 2*rv_std, rv_m + 1*rv_std)

    plt.plot(fv, label='factor_values')
    plt.axhline(factor_buy_hold_interval[0], color='red', linestyle='--', label='Mean + 3*Std')
    plt.axhline(factor_buy_hold_interval[1], color='red', linestyle='--', label='Mean - 1*Std')
    plt.axhline(factor_sell_hold_interval[0], color='green', linestyle='--', label='Mean - 3*Std')
    plt.axhline(factor_sell_hold_interval[1], color='green', linestyle='--', label='Mean + 1*Std')
    plt.legend()
    plt.show()
    plt.plot(rv, label='return_value')
    plt.axhline(return_buy_hold_interval[0], color='red', linestyle='--', label='Mean + 3*Std')
    plt.axhline(return_buy_hold_interval[1], color='red', linestyle='--', label='Mean - 1*Std')
    plt.axhline(return_sell_hold_interval[0], color='green', linestyle='--', label='Mean - 3*Std')
    plt.axhline(return_sell_hold_interval[1], color='green', linestyle='--', label='Mean + 1*Std')
    plt.legend()
    plt.show()

    # 开始测试F1 score方法。
    t_f1 = TimeZone_F1(fv, rv, 2, -1, -2, 1)
    print(t_f1.calculate_F1())