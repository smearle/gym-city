import imutils
from imutils import paths
import os
import shutil

class GifWriter(object):
    def __init__(self):
        self.n_gifs = 0
        self.max_gifs = 10
        self.done = False
    def create_gif(self, input_path, output_path, delay, finalDelay, loop):
            if self.done:
                return
            if self.n_gifs > self.max_gifs:
                shutil.rmtree(input_path)
                self.done = True
                return
            print('creating gif using images in: {}'.format(input_path))
            output_path = os.path.join(output_path, 'test_{}.gif'.format(self.n_gifs))
            # grab all image paths in the input directory
            image_paths = sorted(list(paths.list_images(input_path)))
            # remove the last image path in the list
            last_path = image_paths[-1]
            image_paths = image_paths[:-1]
           #print('ordered_frames:')
           #print(*image_paths, sep='\n')
            # construct the image magick 'convert' command that will be used
            # generate our output GIF, giving a larger delay to the final
            # frame (if so desired)
            print('saving gif at {}'.format(output_path))
            cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
                    delay, " ".join(image_paths), finalDelay, last_path, loop,
                    output_path)
            os.system(cmd)
            self.n_gifs += 1

