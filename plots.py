import matplotlib.pyplot as plt
import csv

filename = 'LunarLander-v2'

#Data colums
episode_counter = []
steps = []
train_counter = []
cumulative_reward = []
average_cumulative_reward = []
learning_flag = []

with open('./training/' + filename + '.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        episode_counter.append(int(row[0]))
        steps.append(int(row[1]))
        train_counter.append(int(row[2]))
        cumulative_reward.append(float(row[3]))
        average_cumulative_reward.append(float(row[4]))
        learning_flag.append(bool(row[5]))

plt.plot(episode_counter, cumulative_reward, label='Cumulative reward')
plt.plot(episode_counter, average_cumulative_reward, label='Last 100 average')
plt.xlabel('Episode')
plt.title('Rewards')
plt.legend()
plt.show()