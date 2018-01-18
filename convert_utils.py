import os
import io
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
from collections import namedtuple
from PIL import Image


def xml_to_groupped_df(xml_examples, label_map_dict):
    assert(len(label_map_dict.keys()) == 1)
    label = list(label_map_dict.keys())[0]

    xml_list = []
    for xml_file in xml_examples:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,      # filename
                     int(root.find('size')[0].text),  # width
                     int(root.find('size')[1].text),
                     label,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            if os.path.splitext(value[0])[1] not in [".jpg", ".jpeg"]:
                print("Skipping not jpeg file:", value[0])
                continue
            xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    data = namedtuple('data', ['filename', 'object'])
    gb = xml_df.groupby("filename")
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, image_dir, label_map_dict):
    image_path = os.path.join(image_dir, '{}'.format(group.filename))
    print("Processing", image_path)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # read all bounding boxes
    it = 1
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])
        print("\t" + "\t".join(["bbox", "xmin", "xmax", "ymin", "ymax", "class", "class_id"]))
        print("\t" + "\t".join(map(str, [it, 
                                         "{0:.2f}".format(xmins[-1]), 
                                         "{0:.2f}".format(xmaxs[-1]), 
                                         "{0:.2f}".format(ymins[-1]), 
                                         "{0:.2f}".format(ymaxs[-1]), 
                                         row['class'], 
                                         classes[-1]]))) 
        it += 1
    print("*" * 60)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """
    print("Trying to save", len(examples), "examples as", output_filename)
    
    writer = tf.python_io.TFRecordWriter(output_filename)
    examples_xml = [os.path.join(annotations_dir, x) + ".xml" for x in examples]
    
    for group in xml_to_groupped_df(examples_xml,label_map_dict):
        tf_example = create_tf_example(group, image_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())
    
    writer.close()
