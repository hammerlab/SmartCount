from cvutils.mrcnn import dataset as mrcnn_dataset


SOURCE = 'celldom'



class CelldomDataset(mrcnn_dataset.RectLabelDataset):

    def initialize(self, image_paths, classes):
        super(CelldomDataset, self).initialize(image_paths, classes, SOURCE)
