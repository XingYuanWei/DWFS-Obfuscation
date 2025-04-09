import sys
import os

def count_files_in_directory(directory):
    """统计指定目录及其子目录下的所有文件数量"""
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在！")
        return 0
    if not os.path.isdir(directory):
        print(f"{directory} 不是一个有效的目录！")
        return 0

    total = 0
    for root, _, files in os.walk(directory):
        total += len(files)
    return total

def main():
    directory = '/storage/xiaowei_data/DWFS-Obfuscation_Data'
    count = count_files_in_directory(directory)
    print(f"目录 {directory} 下共有 {count} 个文件。")

if __name__ == "__main__":
    main()