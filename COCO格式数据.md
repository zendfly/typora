# COCO格式数据

关键字：

- iscrowd：0 or 1，
- segmentation：取决于样本是单个单个对象还是一组对象。单个对象，iscrowd=1，使用polygons格式；一组对象，iscrowd=1，使用RLE格式。

单个的对象可能需要多个polygon来表示，例如对象在图中被挡住了。而iscrowd=1时（将标注一组对象，比如一群人）的segmentation使用就是RLE格式。

polygons，即用一组多边形来标记对象。在coco中，polygons使用4组坐标（矩形）来组成segmentation值。