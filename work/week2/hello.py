import gdown
file_id = 'd/1flNaOwuTgL1Kxj3Q2tpg-QJljvukO6'
#url1 = f'https://drive.google.com/uc?id={file_id}'
url = f'https://drive.google.com/file/d/1flNaOwuTgL1Kxj3Q2tpg-QJljvukO6aq/view?pli=1'
output = 'week2_class2.mp4'

gdown.download(url, output, quiet=False)