import os


def rename_video():
    path = 'dataset'
    files = os.listdir(path)
    for i, file in enumerate(files):
        videos = os.listdir(path + "/" + file)
        for j, video in enumerate(videos):
            new_name = path + "/" + file + "/" + file + "_{}.mp4".format(j)
            old_name = os.path.join(path + "/" + file, video)
            os.rename(old_name, new_name)


def generate_list(list_file: None):
    if list_file is not None:
        path = 'dataset'
        files = os.listdir(path)
        source = []
        label = []
        for i, file in enumerate(files):
            videos = os.listdir(path + "/" + file)
            for j, video in enumerate(videos):
                name = os.path.join(path + "/" + file, video)
                source.append(name)
                label.append(i)
        return source, label

    else:
        return None


if __name__ == '__main__':
    rename_video()
