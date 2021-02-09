import os
import shutil

def moveAllFilesinDir(src_dir,dst_dir):
    if os.path.isdir(src_dir) and os.path.isdir(dst_dir):
        for dir in os.listdir(src_dir):
            shutil.move(os.path.join(src_dir,dir),dst_dir)


def rm_redunct_label(path):
    for cwd, dir, files in os.walk(path):
        if "masks" in cwd:
            for file in files:
                if not "labelIds.png" in file:
                    os.remove(os.path.join(cwd, file))

if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        os.chdir("./data")

    os.mkdir("train"), os.mkdir("train/images"), os.mkdir("train/masks");
    os.mkdir("test"), os.mkdir("test/images"), os.mkdir("test/masks");

    src_dir = "./gtFine_trainvaltest/gtFine/train"
    dst_dir= "./train/masks"
    moveAllFilesinDir(src_dir,dst_dir)

    src_dir = "./leftImg8bit_trainvaltest/leftImg8bit/train"
    dst_dir= "./train/images"
    moveAllFilesinDir(src_dir,dst_dir)


    src_dir = "./gtFine_trainvaltest/gtFine/val"
    dst_dir= "./test/masks"
    moveAllFilesinDir(src_dir,dst_dir)

    src_dir = "./leftImg8bit_trainvaltest/leftImg8bit/val"
    dst_dir= "./test/images"
    moveAllFilesinDir(src_dir,dst_dir)

    shutil.rmtree("gtFine_trainvaltest")
    shutil.rmtree("leftImg8bit_trainvaltest")

    rm_redunct_label(".")

