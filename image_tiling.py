"""Tile imagery in RAM for use in convolutional neural nets."""
import numpy as np
import imageio


class Image_tiler:
    """Tile imagery in RAM for use in convolutional neural nets."""

    def __init__(self, in_arr, out_shape=(256, 256),
                 out_overlap=128, offset=(0, 0)):
        """
        Initialize the class with required data.

        in_arr:         the numpy array to tile
        out_shape:      tuple of (height, width) of resulting tiles
        out_overlap:    int, number of pixels to overlap by
        offset:         tuple, offset from top left corner in pixels
        """
        self.in_arr = in_arr
        self.in_shape = in_arr.shape
        self.out_shape = out_shape
        self.out_overlap = out_overlap
        self.offset = offset

    def crop_image(self):
        """Crops the input image in order to be tileable."""
        height = self.out_shape[0] + self.offset[0]
        while True:
            height += (self.out_shape[0] - self.out_overlap)
            if self.in_shape[0] < height:
                height -= (self.out_shape[0] - self.out_overlap)
                break

        width = self.out_shape[1] + self.offset[1]
        while True:
            width += (self.out_shape[1] - self.out_overlap)
            if self.in_shape[1] < width:
                width -= (self.out_shape[1]-self.out_overlap)
                break

        self.crop_arr = self.in_arr[self.offset[0]:height,
                                    self.offset[1]:width, :]
        return self.crop_arr

    def tile_image(self):
        """Tiles the input image in order to use it in CNNs."""
        self.tiles_num_ver = int((self.in_shape[0] - self.out_shape[0])
                                 / (self.out_shape[0] - self.out_overlap)) + 1
        self.tiles_num_hor = int((self.in_shape[1] - self.out_shape[1])
                                 / (self.out_shape[1] - self.out_overlap)) + 1

        tiles_num = self.tiles_num_ver * self.tiles_num_hor
        self.tiles_arr = np.empty((tiles_num, self.out_shape[0],
                                  self.out_shape[1], self.in_shape[2]),
                                  self.in_arr.dtype)
        idx = 0
        for row in range(self.tiles_num_ver):
            for col in range(self.tiles_num_hor):
                row_start = row * (self.out_shape[0] - self.out_overlap)
                col_start = col * (self.out_shape[1] - self.out_overlap)
                self.tiles_arr[idx, :, :, :] = self.in_arr[
                    row_start:row_start+self.out_shape[0],
                    col_start:col_start+self.out_shape[1], :]
                idx += 1
        return self.tiles_arr


if __name__ == '__main__':
    dummy_filename = 'C:\\Users\\dd\\Pictures\\DSC_0084.jpg'
    dummy_arr = imageio.imread(dummy_filename)  # .astype(np.float32)

    print(dummy_arr.dtype)
    print(dummy_arr.shape)

    dummy_dataset = Image_tiler(dummy_arr, (256, 256), 128, (0, 0))
    dummy_crop = dummy_dataset.crop_image()
    print(dummy_crop.shape)
    dummy_tiles = dummy_dataset.tile_image()
    print(dummy_tiles.shape)
