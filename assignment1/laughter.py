import pandas as pd 
import numpy as np

def independent_t_test(data1, data2):
    """
    Perform independent sample t-test

    Parameters:
    - Data1: First set of data
    - Data2: Second set of data

    Return value:
    - T_ Stat: t-statistic
    - Df: degrees of freedom
    """

    mean1 = sum(data1) / len(data1)
    mean2 = sum(data2) / len(data2)

    std1 = (sum((x - mean1) ** 2 for x in data1) / len(data1)) ** 0.5
    std2 = (sum((x - mean2) ** 2 for x in data2) / len(data2)) ** 0.5

    df = len(data1) + len(data2) - 2

    t_stat = (mean1 - mean2) / ((std1**2 / len(data1)) + (std2**2 / len(data2))) ** 0.5
    return t_stat, df

def independent_chi2_test(observed_data1, observed_data2, expected_data1, expected_data2):
    """
    Perform Chi Square Probabilities test

    Parameters:
    - Observed_ Data1: Frequency of the first set of data
    - Observed_ Data2: Second set of data frequency
    - Expected_ Data1: Expected frequency of the second set of data
    - Expected_ Data2: Expected frequency of the second set of data

    Return value:
    - Chi_ Squared_ Stat: chi_ Squared_ Stat statistic
    - Df: degrees of freedom
        """

    observed_data = np.array([observed_data1, observed_data2])
    expected_data = np.array([expected_data1, expected_data2])

    df = (2-1) * (2-1)

    chi_squared_stat = np.sum((observed_data - expected_data)**2 / expected_data)
    return chi_squared_stat, df


laughter = pd.read_csv('./laughter-corpus.csv') # dataframe



# 1
laughter_female = laughter.loc[laughter['Gender']=='Female']
laughter_male = laughter.loc[laughter['Gender']=='Male']

print("Number of laughter events of female" + str(len(laughter_female)))
print("Number of laughter events of male" + str(len(laughter_male)))

observed_female = len(laughter_female)
observed_male = len(laughter_male)
 
expected_female = 63 / 120 * (len(laughter_male) + len(laughter_female))
expected_male = 57 / 120 * (len(laughter_male) + len(laughter_female))

chi_squared_stat, df = independent_chi2_test(observed_female, observed_male, expected_female, expected_male)

print(chi_squared_stat)
if chi_squared_stat > 3.84: # p < 0.05
    reject_null = True
    print("Reject null hypothesis: The number of laughter events differs between Male and Female.")
    print("Confidence level: 95%")
else:
    reject_null = False
    print("Accepet null hypothesis: The number of laughter events are same between Male and Female.")
    print("Confidence level: 95%")

# 2
laughter_caller = laughter.loc[laughter['Role']=='Caller']
laughter_receiver = laughter.loc[laughter['Role']=='Receiver']

print("Number of laughter events of caller" + str(len(laughter_caller)))
print("Number of laughter events of reveiver" + str(len(laughter_receiver)))

observed_caller = len(laughter_caller)
observed_receiver= len(laughter_receiver)

expected_caller = 0.5 * (len(laughter_caller) + len(laughter_receiver))
expected_receiver = 0.5 * (len(laughter_caller) + len(laughter_receiver))

chi_squared_stat, df = independent_chi2_test(observed_caller, observed_receiver, expected_caller, expected_receiver)
print(chi_squared_stat)
if chi_squared_stat > 3.84: # p < 0.05
    reject_null = True
    print("Reject null hypothesis: The number of laughter events differs between Callers and Receivers.")
    print("Confidence level: 95%")
else:
    reject_null = False
    print("Confidence level: 95%")

# 3
durations_female = laughter.loc[laughter['Gender']=='Female']['Duration']
durations_male = laughter.loc[laughter['Gender']=='Male']['Duration']

mean_female_duration = durations_female.values.mean()
mean_male_duration = durations_male.values.mean()

print("Mean value of Female laughter events: "+str(mean_female_duration))
print("Mean value of Male laughter events: "+str(mean_male_duration))

std_female_duration = durations_female.values.std()
std_male_duration = durations_male.values.std()

print("Std value of Female laughter events: "+str(std_female_duration))
print("Std value of Male laughter events: "+str(std_male_duration))


t_stat, df = independent_t_test(durations_female, durations_male)
print(t_stat)
if t_stat > 1.960: # p < 0.05
    reject_null_gender = True
    print("Reject null hypothesis: The duration of laughter events longer for women.")
    print("Confidence level: 95%")
else:
    reject_null_gender = False
    print("Confidence level: 95%")


# 4
durations_caller = laughter.loc[laughter['Role']=='Caller']['Duration'].astype(float)
durations_receiver = laughter.loc[laughter['Role']=='Receiver']['Duration'].astype(float)

mean_caller_duration = durations_caller.values.mean()
mean_receiver_duration = durations_receiver.values.mean()

print("Mean value of Caller laughter events: "+str(mean_caller_duration))
print("Mean value of Receiver laughter events: "+str(mean_receiver_duration))

std_caller_duration = durations_caller.values.std()
std_receiver_duration = durations_receiver.values.std()

print("Std value of Caller laughter events: "+str(std_caller_duration))
print("Std value of Receiver laughter events: "+str(std_receiver_duration))

t_stat, df = independent_t_test(durations_caller, durations_receiver)
print(t_stat)
if t_stat > 1.960: # p < 0.05
    reject_null_call = True
    print("Reject null hypothesis: The duration of laughter events is longer for callers")
    print("Confidence level: 95%")
else:
    reject_null_call = False
    print("Confidence level: 95%")

