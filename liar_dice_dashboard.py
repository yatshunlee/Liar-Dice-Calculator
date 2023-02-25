# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from math import comb
from typing import List
from scipy.stats import binom
from collections import Counter
import matplotlib.pyplot as plt

class DiceCup:
    def __init__(self, dices:List[int], num_players:int):
        if dices:
            self.dices = Counter(sorted(dices)) # counter dict
            self.num_players = num_players
            self.num_dices = 5 * num_players
    
    def get_kdot(self, n:int, k:int, isPure:bool=False) -> int:
        """
        to get the remaining number of dices required to fulfill the call
        :param: n: e.g. n of a kind e.g. f([3, 3, 3, 4, 4]) = 2's 4 where n = 2
        :param: k: e.g. n of a kind e.g. f([3, 3, 3, 4, 4]) = 2's 4 where k = 4
        :param: isPure: if not pure, 1 can be counted as anything; else, cannot
        :return: kdot: person's call - what you have in hand
        """
        yours = self.dices[k]
        if not isPure:
            yours += self.dices[1]
        if yours == 5:
            yours += 1
        return n - yours
    
    def eval_call(self, n:int, k:int, isPure:bool=False, believe:bool=True) -> float:
        """
        to calculate the win rate of the call
        
        :param: n: e.g. n of a kind e.g. f([3, 3, 3, 4, 4]) = 2's 4 where n = 2
        :param: k: e.g. n of a kind e.g. f([3, 3, 3, 4, 4]) = 2's 4 where k = 4
        :param: isPure: if not pure, 1 can be counted as anything; else, cannot
        :param: believe: assumption based on expected value (subjective info)
        :return: win rate (bounded by 0.0 to 1.0)
        """
        theta = 2 / 6 if not isPure else 1 / 6
        
        kdot = self.get_kdot(n, k, isPure)
        
        # return if you don't need extra (from others)
        # let's say someone asking for 5'6 and you have it in your dice cup
        if kdot <= 0:
            return 1.0
        
        # return the conditional probability based on the assumption
        if believe:
            num_dices = self.num_dices - 5 * 2
            
            # P(Y>=1)
            if isPure:
                p_player_has = 1 - binom.pmf(k=0, n=5, p=theta)
            # P(Y>=2)
            else:
                p_player_has = 1 - binom.pmf(k=0, n=5, p=theta) - binom.pmf(k=1, n=5, p=theta)
            
            # P(X>=k) * P(Y=2) + ... + P(X>=k) * P(Y=5), where X = K + Y
            p_joint = 0
            start = 1 if isPure else 2
            for j in range(start, 6):
                p_joint += binom.pmf(k=j, n=5, p=theta) * (
                    1 - binom.cdf(k=kdot-1-j, n=num_dices, p=theta)
                )
            return p_joint / p_player_has
        
        # return a general one
        else:
            # 1 - P(X<k) given that you know the dices in your hand
            num_dices = self.num_dices - 5
            return 1 - binom.cdf(k=kdot-1, n=num_dices, p=theta)
        
    def plot_heatmap(self, isPure:bool=False, believe:bool=False):
        """heatmap of win rate"""
        prob_heatmap = []
        
        if isPure:
            occurance = range(self.num_players, self.num_dices+1)
        else:
            occurance = range(self.num_players+2, self.num_dices+1)
        dice_num = range(1, 7)
        
        max_count = 0
        for k in dice_num:
            # when not pure: dont need to calculate 1
            if k == 1 and not isPure:
                prob_row = [0] * len(occurance)
                
            else:
                prob_row = []
                for i in occurance:
                    # when k != 1 and first col = 0 => p = 0
                    if i == occurance[0] and k != 1 and isPure:
                        p = 0
                    else:
                        p = self.eval_call(i, k, isPure, believe)
                    prob_row.append(p)
            
            prob_heatmap.append(prob_row)
            
            # trim the redundant part of the heatmap
            count = 0
            for p in prob_row:
                if p >= 0.005:
                    count += 1
            max_count = max(count, max_count)
            
        for i in range(len(prob_heatmap)):
            prob_heatmap[i] = prob_heatmap[i][:max_count]
        
        fig, ax = plt.subplots(figsize=(2*self.num_players, 6))
        sns.heatmap(
            prob_heatmap, annot=True, fmt='.2f',
            xticklabels=occurance[:max_count],
            yticklabels=dice_num,
            ax=ax
        )
        ax.set_xlabel('call')
        ax.set_ylabel('dice')
        return fig
    
    def plot_each_opponent_pmf(self, isPure:bool=False):
        """plot occurance of each individual player"""
        theta = 2 / 6 if not isPure else 1 / 6
        prob_dist = []
        for k in range(6):
            p = binom.pmf(k, 5, theta)
            prob_dist.append(p)
        
        fig, ax = plt.subplots()
        ax.bar(range(6), prob_dist)
        ax.set_ylabel('prob')
        ax.set_xlabel('number of the same dice')
        pure_text = 'Pure' if isPure else 'notPure'
        exp_val = sum([i*p if i != 5 else 6*p for i, p in enumerate(prob_dist)])
        ax.set_title("Expected value ({}): {:.4f}".format(pure_text, exp_val))
        return fig


def main():
    # Set page title
    st.set_page_config(page_title='Data Plot Dashboard')

    # Set sidebar title and widgets
    st.sidebar.title('Data Input')
    num_people = st.sidebar.number_input('Number of People', min_value=1, max_value=10, value=5)
    dices = st.sidebar.text_input('Your dices (comma-separated integers)')
    isPure = st.sidebar.radio('Pure or not', ('Not', 'Pure'))
    isBelieve = st.sidebar.radio('Do you believe', ('Not', 'Believe'))
    go = st.sidebar.button('Analyze')

    # Parse comma-separated string into list of integers
    if dices:
        dices = list(map(int, dices.split(',')))
    else:
        dices = None

    dc = DiceCup(dices, 6)
    
    # Plot data and display on dashboard
    if dices:
        isPure = True if isPure == 'Pure' else False
        isBelieve = True if isBelieve == 'Believe' else False
        fig1 = dc.plot_heatmap(isPure, isBelieve)
        st.pyplot(fig1)


if __name__ == '__main__':
    main()
