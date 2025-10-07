# import requests
# import tarfile
# from tqdm import tqdm
# from huggingface_hub import hf_hub_url

# # --- 配置参数 ---
# REPO_ID = "TencentARC/StoryStream"  # 数据集仓库ID
# ARCHIVE_FILENAME = "George/george.tar.gz"   # 你要下载的压缩包文件名
# TARGET_IMAGE_NAME = "000123/000123_keyframe_0-14-28-659.jpg" # 示例：你想要提取的图片在压缩包内的路径
# OUTPUT_FILENAME = "downloaded_image.jpg" # 保存到本地的文件名

# # 1. 从 Hugging Face Hub 获取文件的真实下载链接
# url = hf_hub_url(repo_id=REPO_ID, filename=ARCHIVE_FILENAME, repo_type="dataset")
# print(f"获取到下载链接: {url}")

# # 2. 使用 requests 库以流式（stream）方式请求文件
# # stream=True 是关键，它不会一次性把所有内容加载到内存
# with requests.get(url, stream=True) as r:
#     r.raise_for_status() # 确保请求成功
    
#     # 获取文件总大小，用于显示进度条
#     total_size_in_bytes = int(r.headers.get('content-length', 0))
    
#     # 创建一个进度条
#     progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="正在扫描压缩包")

#     # 3. 将下载流直接喂给 tarfile 库进行解压
#     # mode='r|gz' 表示以流式读取 gzip 压缩的 tar 文件
#     with tarfile.open(fileobj=r.raw, mode='r|gz') as tar:
#         print("成功打开压缩包流，开始查找目标文件...")
#         for member in tar:
#             # 更新进度条，这里只能模拟一个大概的进度
#             # 因为我们不知道目标文件在压缩包的哪个位置
#             if r.raw.tell() < total_size_in_bytes:
#                 progress_bar.update(r.raw.tell() - progress_bar.n)

#             # 4. 检查当前文件是不是你想要的那个
#             if member.name == TARGET_IMAGE_NAME:
#                 print(f"\n找到了目标文件: {member.name}")
                
#                 # 5. 提取文件内容
#                 f = tar.extractfile(member)
#                 if f is not None:
#                     # 6. 将文件内容保存到本地
#                     with open(OUTPUT_FILENAME, 'wb') as outfile:
#                         outfile.write(f.read())
#                     print(f"图片已成功保存为: {OUTPUT_FILENAME}")
#                     # 7. 找到文件后，跳出循环
#                     break
#         else:
#              # 如果循环正常结束（没有break），说明没找到文件
#              print(f"压缩包扫描完毕，未找到文件: {TARGET_IMAGE_NAME}")

#     # 脚本执行到这里会自动关闭连接，从而停止下载
#     progress_bar.close()
#     if progress_bar.n < total_size_in_bytes and 'f' in locals():
#          print("已停止下载，节省了大量带宽。")
#     else:
#          print("扫描已完成。")

import requests
import tarfile
from tqdm import tqdm
from huggingface_hub import hf_hub_url
import os

# --- 配置参数 ---
REPO_ID = "TencentARC/StoryStream"  # 数据集仓库ID
ARCHIVE_FILENAME = "George/george.tar.gz"   # 压缩包文件名
DOWNLOAD_LIMIT = 10                   # 新增：设置下载图片的数量上限
OUTPUT_DIRECTORY = "downloaded_images"  # 新增：所有图片将保存在这个文件夹

# --- 脚本开始 ---

# 创建输出文件夹（如果它不存在）
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# 1. 从 Hugging Face Hub 获取文件的真实下载链接
url = hf_hub_url(
    repo_id=REPO_ID, 
    filename=ARCHIVE_FILENAME, 
    repo_type="dataset",
    # endpoint="https://huggingface.co" # 使用官方源
)
print(f"获取到官方下载链接: {url}")

# (可选) 如果你的服务器需要代理，请取消下面的注释并配置
# proxies = {
#    'http': 'http://127.0.0.1:7890',
#    'https': 'http://127.0.0.1:7890',
# }

images_downloaded_count = 0 # 新增：已下载图片的计数器

# 2. 使用 requests 库以流式（stream）方式请求文件
with requests.get(url, stream=True) as r: # 如果需要代理，加上 proxies=proxies
    r.raise_for_status()
    
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="正在扫描压缩包")

    # 3. 将下载流直接喂给 tarfile 库进行解压
    with tarfile.open(fileobj=r.raw, mode='r|gz') as tar:
        print(f"成功打开压缩包流，开始查找并下载前 {DOWNLOAD_LIMIT} 张图片...")
        for member in tar:
            # 更新进度条
            if r.raw.tell() < total_size_in_bytes:
                progress_bar.update(r.raw.tell() - progress_bar.n)

            # 4. 检查当前成员是否是一个文件，并且是图片格式
            if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                # 5. 提取文件内容
                f = tar.extractfile(member)
                if f is not None:
                    # 成功找到一张图片，计数器加一
                    images_downloaded_count += 1
                    
                    # 6. 创建带编号的文件名，并指定保存路径
                    base_filename = os.path.basename(member.name)
                    # 格式化文件名，如 01_xxx.jpg, 02_yyy.jpg ...
                    output_filename = f"{images_downloaded_count:02d}_{base_filename}"
                    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
                    
                    # 7. 将文件内容保存到本地
                    with open(output_path, 'wb') as outfile:
                        outfile.write(f.read())
                    print(f"已下载: {output_path} (第 {images_downloaded_count}/{DOWNLOAD_LIMIT} 张)")
                    
                    # 8. 检查是否已达到下载上限
                    if images_downloaded_count >= DOWNLOAD_LIMIT:
                        print(f"\n已成功下载 {DOWNLOAD_LIMIT} 张图片，任务完成。")
                        break # 达到上限，跳出循环，停止下载
        else:
             # 如果循环正常结束（没有break），说明压缩包已扫描完但未满10张
             print(f"压缩包扫描完毕。共下载了 {images_downloaded_count} 张图片。")

    progress_bar.close()
    if images_downloaded_count >= DOWNLOAD_LIMIT:
         print("已停止下载，节省了大量带宽。")
    else:
         print("扫描已完成或提前终止。")