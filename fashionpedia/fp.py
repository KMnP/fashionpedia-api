"""
API for accessing Fashionpedia Dataset in JSON format.
FASHIONPEDIA API is a Python API that assists in loading, parsing and visualizing
the annotations in Fashionpedia.
"""
import os
import sys
import time
import json
import copy
from collections import defaultdict
import skimage.io as io
import matplotlib.pyplot as plt

import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
PYTHON_VERSION = sys.version_info[0]


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def _isInList(input, target):
    for i in input:
        if i in target:
            return True
    return False


class Fashionpedia(COCO):
    def __init__(self, annotation_file=None):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        super(Fashionpedia, self).__init__()
        if annotation_file:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, \
                'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        super(Fashionpedia, self).createIndex()
        # attributes
        attrs = {}
        attToImgs = defaultdict(list)
        print("loading attributes...")
        if 'attributes' in self.dataset:
            for att in self.dataset['attributes']:
                attrs[att['id']] = att

        if 'annotations' in self.dataset and 'attributes' in self.dataset:
            for ann in self.dataset['annotations']:
                attToImgs[ann['category_id']].append(ann['image_id'])
        print("attributes index created!")
        self.attToImgs = attToImgs
        self.attrs = attrs

    def getAnnIds(
        self,
        imgIds=[],
        catIds=[],
        areaRng=[],
        attIds=[],
    ):
        """
        Get ann ids that satisfy given filter conditions.
        default skips that filter
        Args:
            imgIds  (int or int array): get anns for given imgs
            catIds  (int or int array): get anns for given atts
            areaRng (float array): get anns for given area range(e.g. [0 inf])
            attIds (int array): get anns for given atts
            iscrowd (boolean): get anns for given crowd label (False or True)
        Returns:
            ids (int array): integer array of ann ids
        """
        annIds = super(Fashionpedia, self).getAnnIds(imgIds, catIds, areaRng)
        attIds = attIds if _isArrayLike(attIds) else [attIds]

        anns = self.loadAnns(annIds)
        anns = anns if len(attIds) == 0 else [ann for ann in anns if _isInList(
            ann["attribute_ids"], attIds)]
        ids = [ann['id'] for ann in anns]
        return ids

    def getAttIds(self, attNms=[], supNms=[], attIds=[]):
        """
        get attribute ids with following filtering parameters.
        default skips that filter.
        Args:
            attNms (str array): get atts for given att names
            supNms (str array): get atts for given supercategory names
            attIds (int array): get atts for given att ids
        Returns:
            ids (int array): integer array of att ids
        """
        attNms = attNms if _isArrayLike(attNms) else [attNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        attIds = attIds if _isArrayLike(attIds) else [attIds]

        if len(attNms) == len(supNms) == len(attIds) == 0:
            atts = self.dataset['attributes']
        else:
            atts = self.dataset['attributes']
            atts = atts if len(attNms) == 0 \
                else [att for att in atts if att['name'] in attNms]
            atts = atts if len(supNms) == 0 \
                else [att for att in atts if att['superattegory'] in supNms]
            atts = atts if len(attIds) == 0 \
                else [att for att in atts if att['id'] in attIds]
        ids = [att['id'] for att in atts]
        return ids

    def getImgIds(self, imgIds=[], catIds=[], attIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        Args:
            imgIds (int array): get imgs for given ids
            catIds (int array): get imgs with all given cats
            attIds (int array): get imgs with all given atts
        Returns:
            ids (int array)  : integer array of img ids
        '''
        ids = super(Fashionpedia, self).getImgIds(imgIds, catIds)
        attIds = attIds if _isArrayLike(attIds) else [attIds]
        for i, attId in enumerate(attIds):
            if i == 0 and len(ids) == 0:
                ids = set(self.attToImgs[attId])
            else:
                ids &= set(self.attToImgs[attId])
        return list(ids)

    def loadAttrs(self, ids=[]):
        """
        Load atts with the specified ids.
        Args:
            ids (int array): integer ids specifying atts
        Returns:
            atts (object array): loaded att objects
        """
        if _isArrayLike(ids):
            return [self.attrs[idx] for idx in ids]
        elif type(ids) == int:
            return [self.attrs[ids]]

    def showAnns(self, anns):
        """addtionally print out the attribute annotations"""
        super(Fashionpedia, self).showAnns(anns)
        # display category and attributes for asscosiated segmentation
        for i, ann in enumerate(anns):
            print("Segmentation {}:".format(i))
            print("\tCategory: {}".format(
                self.cats[ann["category_id"]]["name"]))
            if len(ann["attribute_ids"]) > 0:
                print("\tAttribtues:")
                for attId in ann["attribute_ids"]:
                    print("\t\t{}: {}".format(
                        self.attrs[attId]["id"], self.attrs[attId]["name"]))

    def visualize(self, imgId, imgRoot, catIds=[], attIds=[]):
        """
        display annotations for one image only.
        Display two image side-by-side:
            left: original image
            right: images with annotated mask with specificed catIds
            print out the attributes at top.
        Args:
            imgId (int): image idx to visualize
            imgRoot (str): path to images
            catIds (List(int)): list of cat to display
            attIds (List(int)): list of att to display
        """
        # load image
        plt.rcParams['figure.figsize'] = [30, 20]
        plt.subplot(1, 2, 1)
        plt.axis('off')
        img = self.loadImgs(imgId)[0]
        imgArray = io.imread(os.path.join(imgRoot, img['file_name']))
        plt.imshow(imgArray)

        plt.subplot(1, 2, 2)
        # load and display instance annotations
        plt.imshow(imgArray)
        plt.axis('off')
        annIds = self.getAnnIds(imgIds=img['id'], catIds=catIds, attIds=attIds)

        anns = self.loadAnns(annIds)
        self.showAnns(anns)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = Fashionpedia()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str \
           or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            anns = json.load(open(resFile))
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'

        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(
                self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if 'segmentation' not in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(
                self.dataset['categories'])
            for idx, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation
                ann['area'] = maskUtils.area(ann['segmentation'])
                if 'bbox' not in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = idx+1
                ann['iscrowd'] = 0

        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.dataset["attributes"] = self.dataset["attributes"]
        res.dataset["categories"] = self.dataset["categories"]
        res.createIndex()
        return res
        # raise NotImplementedError

    def download(self, tarDir=None, imgIds=[]):
        raise NotImplementedError

    def loadNumpyAnnotations(self, data):
        raise NotImplementedError
