import csv

# 从CSV文件读取数据
data = []
with open('/media/user/sda13/home/jli/DeepGS_new/cow/FDG/FDG_snp_all_modified_SNP.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

# 转置数据，将行变为列
transposed_data = list(map(list, zip(*data)))

# 初始化频数为0的列表
counts_A = [0] * len(transposed_data)
counts_a = [0] * len(transposed_data)

# 遍历每个子列表，统计频数
for i, sublist in enumerate(transposed_data):
    for item in sublist:
        if isinstance(item, str):
            counts_A[i] += item.count('A')
            counts_a[i] += item.count('a')

# 替换每一列中的AA、Aa和aa值
for i in range(len(transposed_data)):
    total_count = counts_A[i] + counts_a[i]
    A = counts_A[i] / total_count
    a = counts_a[i] / total_count
    Aa = A * a
    AA = A * A
    aa = a * a

    for j in range(len(transposed_data[i])):
        if transposed_data[i][j] == 'AA':
            transposed_data[i][j] = str(AA)
        elif transposed_data[i][j] == 'Aa':
            transposed_data[i][j] = str(Aa)
        elif transposed_data[i][j] == 'aa':
            transposed_data[i][j] = str(aa)

# 转置回原始数据，将列变为行
modified_data = list(map(list, zip(*transposed_data)))

# 将修改后的数据保存到CSV文件
output_file = '/media/user/sda13/home/jli/DeepGS_new/cow/FDG/FDG_snp_all_modified_SNP_freq.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(modified_data)

print(f"修改后的数据已保存到文件: {output_file}")