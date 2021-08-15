import csv

import glob2

file_name = "data.csv"
fields = ['Id', 'Genre', 'GiaDinh', 'GiaThong', 'HocDuong', 'TichCuc', 'TieuCuc', 'TrungTinh']
exts=['.jpg', '.jpeg', '.png', '.jfif']
genre_col = []
with open(file_name, 'w') as data_file:
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(fields)
    list_images = glob2.glob(r'Dataset' + '/**')
    image_links = []
    for image_link in list_images:
        for ext in exts:
            if ext in image_link[-5:]:
                image_links.append(image_link)
    imagePaths = sorted(image_links)
    labels = [path.split("\\")[1] for path in imagePaths]
    for i in range(len(imagePaths)):
        genre, sentiment = labels[i].split("_")
        genre_col.append([genre, sentiment])
        if genre == 'GiaDinh' and sentiment == 'Tichcuc':
            row = [imagePaths[i], genre_col, 1, 0, 0, 1, 0, 0]
            csv_writer.writerow(row)
        if genre == 'GiaDinh' and sentiment == 'Tieucuc':
            row = [imagePaths[i], genre_col, 1, 0, 0, 0, 1, 0]
            csv_writer.writerow(row)
        if genre == 'GiaDinh' and sentiment == 'Trungtinh':
            row = [imagePaths[i], genre_col, 1, 0, 0, 0, 0, 1]
            csv_writer.writerow(row)
        if genre == 'GiaoThong' and sentiment == 'Tichcuc':
            row = [imagePaths[i], genre_col, 0, 1, 0, 1, 0, 0]
            csv_writer.writerow(row)
        if genre == 'GiaoThong' and sentiment == 'Tieucuc':
            row = [imagePaths[i], genre_col, 0, 1, 0, 1, 0, 0]
            csv_writer.writerow(row)
        if genre == 'GiaoThong' and sentiment == 'Trungtinh':
            row = [imagePaths[i], genre_col, 0, 1, 0, 1, 0, 0]
            csv_writer.writerow(row)
        if genre == 'HocDuong' and sentiment == 'Tichcuc':
            row = [imagePaths[i], genre_col, 0, 0, 1, 1, 0, 0]
            csv_writer.writerow(row)
        if genre == 'HocDuong' and sentiment == 'Tieucuc':
            row = [imagePaths[i], genre_col, 0, 0, 1, 0, 1, 0]
            csv_writer.writerow(row)
        if genre == 'HocDuong' and sentiment == 'Trungtinh':
            row = [imagePaths[i], genre_col, 0, 0, 1, 0, 0, 1]
            csv_writer.writerow(row)
        genre_col = []