import os

image_extension = ["jpg", "png", "gif", "bmp", "jpeg", "tif", "tiff", "jif", "jfif", "jp2", "jpx", "j2k", "j2c", "fpx", "pcd"]


def get_extension(i):
    ext = os.path.splitext(i)[-1].lower()
    if ext[1:len(ext)] in image_extension:
        return ext[1:len(ext)]
    else:
        return False


def image_file_in_folder(folder_path, flag):
    file_list = []
    file_name = []

    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        for i in filenames:
            try:
                ext = get_extension(i)
                if ext is not False:
                    file_list += [dirpath + os.sep + i]
                    file_name += [i]

            except:
                continue
        if flag == 0:
            break
    return [file_list, file_name] # , file_extension]


def get_system_metadata(folder_path, flag=1):
    [file_list, file_name] = image_file_in_folder(folder_path, flag)
    f = open('file_list.txt', 'w')
    file_size = []
    for i in range(len(file_list)):
        file_size += [os.stat(file_list[i]).st_size]
        f.write(file_list[i] + '\n')
    f.close()
    return [file_list, file_name] # , file_extension, file_size]
