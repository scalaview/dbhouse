order = st.arma_order_select_ic(dta,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
order.bic_min_order
(4, 2)


arma_mod30 = sm.tsa.ARMA(dta, (4, 2)).fit(disp=False)
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
2575.3775921542815 2605.2443223694636 2587.3183678447735
# 分别比较多个模型的AIC, BIC， 两者越小越好
sm.stats.durbin_watson(arma_mod30.resid.values)

D.W统计量是用来检验残差分布是否为正态分布的，因为用OLS进行回归估计是假设模型残差服从正态分布的，因此，如果残差不服从正态分布，那么，模型将是有偏的，也就是说模型的解释能力是不强的。
D.W统计量在2左右说明残差是服从正态分布的，若偏离2太远，那么你所构建的模型的解释能力就要受影响了。




白噪声时间序列的定义是均值为零，方差恒定和相关性为零。
如果你的时间序列是白噪声，那么它无法进行预测。否则，你可能可以改善这个模型。
你可以在时间序列上使用统计数据和诊断图，用以检查它是否是白噪声。



stats.normaltest(resid)
NormaltestResult(statistic=49.845019661107585, pvalue=1.5006917858823576e-11)
# 传统上，在统计学上，你需要一个小于0.05的p值来拒绝零假设。


acorr_ljungbox(resid, lags=1)
(array([0.01404061]), array([0.90567698]))
# p值远小于0.05，平稳非白噪声序列，反之为白噪声序列