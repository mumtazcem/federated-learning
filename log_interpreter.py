import experiment_manager as em
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

result_path = "/Users/mumtazcemeris/PycharmProjects/federated-learning/results/trash/"
filename = "xp_82681.npz"
saved_filename = "saved_results.npz"

# save_path = "/Users/mumtazcemeris/PycharmProjects/federated-learning/results/experiment1/xp_69027.npz"
# balancedness = 1.0
# save_path = "/Users/mumtazcemeris/PycharmProjects/federated-learning/results/experiment1/xp_2096.npz"
# balancedness = 0.5
save_path = "/Users/mumtazcemeris/PycharmProjects/federated-learning/results/experiment1/xp_68347.npz"
balancedness = 2
clientnumber= 10

# data = np.load(result_path+filename, allow_pickle=True)
data = np.load(save_path, allow_pickle=True)
lst = data.files

computing_weights = "client" + "0" + "_compute_weight_update at "
computing_synch = "client" + "0" + "_synchronize_with_server at "
aggregate = "aggregate_weight_updates at "

w = {}
s = {}
a = {}
for i in range(0,clientnumber):
    computing_weights = "client" + str(i) + "_compute_weight_update at "
    computing_synch = "client" + str(i) + "_synchronize_with_server at "
    for item in lst:
        if str(item) == computing_weights:
            print(data[item])
            w[str(i)] = data[item]
        if str(item) == computing_synch:
            print(data[item])
            s[str(i)] = data[item]
        if str(item) == aggregate:
            print(data[item])
            a['aggregate_time'] = data[item]

print(w)
print(s)
print(a)

# ids = range(0, clientnumber)
# values = []
# for i in range(0, clientnumber):
#     values.append(len(w[str(i)]))
#
# plt.bar(ids, values)
# plt.xlabel("Client IDs")
# plt.ylabel("Frequency for contribution")
# plt.title("Client contribution frequencies in FedAvg")
# plt.savefig('figures/freq.png')
# plt.show()
data = []
for item in w:
    print(item, w[item])
    for timestamp in w[item]:
        data.append(["client" + item, timestamp])
for timestamp in a['aggregate_time']:
    data.append(["server", timestamp])
df = pd.DataFrame(data, columns=['participant', 'timeStamp'])
print(data)

row, col = df.shape
x_axis = range(0, row)
df.sort_values(by=['timeStamp'], inplace=True, ascending=True)
df = df.reset_index(drop=True)
df_client = df[df['participant'] != "server"]
df_server = df[df['participant'] == "server"]
plt.figure(figsize=(15, 10))
plt.scatter(df_client.index, df_client['timeStamp'], color='blue', label='Clients')
plt.scatter(df_server.index, df_server['timeStamp'], color='red', label='Server')
plt.xlabel("Arrival")
plt.ylabel("TimeStamp")
plt.title("Weight Updates and Server Aggregations in FedAvg")
plt.savefig('figures/interarrival' + '_balance' + str(balancedness) + '.png')
plt.show()

df_small_client = df[(df['timeStamp'] < 25) & (df['participant'] != "server")]
df_small_server = df[(df['timeStamp'] < 25) & (df['participant'] == "server")]
plt.scatter(df_small_client.index, df_small_client['timeStamp'], color='blue', label='Clients')
plt.scatter(df_small_server.index, df_small_server['timeStamp'], color='red', label='Server')
plt.xlabel("Arrival")
plt.ylabel("TimeStamp")
plt.legend()
plt.title("Weight Updates and Server Aggregations in FedAvg")
plt.savefig('figures/interarrival_small' + '_balance' + str(balancedness) + '.png')
plt.show()


# results_dict = em.load_results(result_path, filename)
# df = pd.DataFrame.from_dict(results_dict)
# df.to_csv('output.csv')
# em.save_results(results_dict, save_path, "saved_results")

