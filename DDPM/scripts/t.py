import numpy as np




# 第1次roll,100%+1
# 第2-9次roll,90%+1， 0.1就停

# 那么，停1的概率=roll两次=1*0.1，停2概率=roll三次=1*0.9*0.1，停3概率=roll四次=1*0.9*0.9*0.1，停N=1*0.9^(N-1)*0.1.
# 10个可能结果, 从停1-停9，是uniform的;停10=all others

limit=10
single_prob=0.9

a=range(1,limit)# 1，2.....9
E=0
prob_cumulative=0

for number in a: # 停1-停9
    this_probability=1*single_prob**(number-1)*(1-single_prob)
    this_value=number
    E+=this_probability*this_value
    prob_cumulative+=this_probability

prob_10=1*single_prob**(limit-1)
assert prob_10+prob_cumulative==1 ,'error'
# 停10=all others
E+=(1-prob_cumulative)*limit

print(E) 

