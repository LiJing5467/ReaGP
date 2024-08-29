import csv

# 读取CSV文件
filename = '/media/user/sda13/home/jli/DeepGS_new/cow/FDG/FDG_geno_snp.csv'
data = []
with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

# 替换字典
replace_dict = {
    0: 'aa',
    1: 'Aa',
    2: 'AA'
}

# 将数据转换为替换后的字符串
converted_data = [[replace_dict[int(value)] for value in row] for row in data]

# 保存到CSV文件
output_filename = '/media/user/sda13/home/jli/DeepGS_new/cow/FDG/FDG_snp_all_modified_SNP.csv'
with open(output_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(converted_data)

print(f"数据已成功转换并保存到文件 {output_filename} 中。")