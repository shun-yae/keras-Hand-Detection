import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

dir_list = "./hand/"
def draw_images(G,x,index,save_dir):
    save_name = "increase" + str(index)
    output_dir = save_dir
    g = G.flow(
            x, batch_size=1, save_to_dir = output_dir,
            save_prefix=save_name, save_format="jpg")
    for i in range(5):
        bach = g.next()

def increase_image(read_dir):
    images = glob.glob(os.path.join(read_dir, "*.jpg"))
    G = ImageDataGenerator(
            rotation_range = 5,
            width_shift_range = 0,
            height_shift_range = 0,
            channel_shift_range = 30.0,
            horizontal_flip = True
            )

    for i in range(len(images)):
        img = load_img(images[i])
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(G,x,i,read_dir)

for i in os.listdir(dir_list):
    increase_image(dir_list + i + "/")
