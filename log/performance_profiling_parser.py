import re
freq_table = []
bw_table = []
time_table = []
file_name = "inceptionv2.txt"
cur_freq = -1
cur_bw = -1
idx_freq = -1
idx_bw = -1
count = 10
with open(file_name, 'r') as file:
    for line in file.readlines():
        if "---Test" in line:
            numbers = [int(s) for s in re.findall(r'\d+', line)]
            cur_freq = numbers[1]
            cur_bw = numbers[2]
            print("cur_freq " + str(cur_freq) + " cur_bw " + str(cur_bw))
            assert (cur_freq > 0)
            assert (cur_bw > 0)
            if cur_freq not in freq_table:
                freq_table.append(cur_freq)
                idx_freq = freq_table.index(cur_freq)
                time_table.append([])
            if cur_bw not in bw_table:
                bw_table.append(cur_bw)
            idx_bw = bw_table.index(cur_bw)
            time_table[idx_freq].append([])
            idx_freq = freq_table.index(cur_freq)
            assert (count == 10)
            count = 0
        elif "Total time - " in line:
            numbers = [float(s) for s in re.findall(r'\d+\.\d+', line)]
            if len(numbers) == 0:
                numbers = [float(s) for s in re.findall(r'\d+', line)]
            time_table[idx_freq][idx_bw].append(numbers[0])
            count += 1

with open("result_" + file_name, 'w') as file:
    file.write(' ' + '\t')
    for j in range(0, len(bw_table)):
        file.write(str(bw_table[j]) + '\t')
    file.write('\n')
    for i in range(0, len(freq_table)):
        cur_freq = freq_table[i]
        file.write(str(cur_freq) + '\t')
        for j in range(0, len(bw_table)):
            cur_bw = bw_table[j]
            print(str(idx_freq) + ' ' + str(idx_bw))
            avg_time = sum(time_table[i][j]) / len(time_table[i][j])
            file.write(str(avg_time) + '\t')
        file.write('\n')
    file.write("\n\n\n\n\n")
    for i in range(0, len(freq_table)):
        cur_freq = freq_table[i]
        for j in range(0, len(bw_table)):
            cur_bw = bw_table[j]
            for num in time_table[i][j]:
                file.write(str(cur_freq) + '\t' + str(cur_bw) + '\t' + str(num) + '\t\n')
