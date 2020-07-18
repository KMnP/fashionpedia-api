"""
evaluation.
"""
import datetime
import numpy as np
import time

from collections import OrderedDict
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import f1_score

from fashionpedia.fp import Fashionpedia


class FPEval(COCOeval):
    """
    Interface for evaluating detection and localized attributes prediction
    on the Fashionpedia dataset.

    The usage for FPEval is as follows:
     fpGt=... fpDt=...            # load dataset and results
     E = FPEval(fpGt,fpDt)        # initialize FpEval object
     E.run()                      # evaluate the results
     E.print_results()            # print results
     E.print_results_iou()        # print result ignoring f1
     E.print_results_f1()         # print result ignoring iou
    See also eval_demo for examples.
    """
    def __init__(self, fpGt=None, fpDt=None, iouType=None):
        '''
        Initialize FPEval using Fashionpeida APIs for gt and dt
        Args:
            fpGt (Fashionpedia object): with ground truth annotations
            fpDt (Fashionpedia object): with detection results
            iouType (str): segm or bbox evaluation
        '''
        if iouType and iouType not in ["segm", "bbox"]:
            raise ValueError(
                "iou type not supported for Fashionpedia evaluation, "
                "got {}".format(iouType))

        if not isinstance(fpGt, Fashionpedia):
            raise ValueError("Groundtruth object type not supported")

        if not isinstance(fpDt, Fashionpedia):
            raise ValueError("Prediction object type not supported")

        super(FPEval, self).__init__(fpGt, fpDt, iouType)
        self.FPParams = FPParams()
        self.results = OrderedDict()
        self.resultDet = OrderedDict()

        self.params.iouThrs = np.append(self.params.iouThrs, -1.0)

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        '''
        super(FPEval, self)._prepare()

        self.cats = self.cocoGt.loadCats(self.params.catIds)
        self.superCats = self.FPParams.catSuperClsIds

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.evaluateImg.
        """
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print(
                "useSegm (deprecated) is not None."
                "Running {} evaluation".format(p.iouType)
            )
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }
        # ignore any categories that is not having any attributes
        self.f1s = {
            (imgId, catId): self.computeF1(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
            if catId in self.FPParams.catsWithAttributes
        }

        # self.gt_attributes_ids = []

        # loop through images, area range, max detection number
        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, p.maxDets[-1])
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        # self._paramsEval = copy.deepcopy(self.params)  # seems do not need it
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _getGtDt(self, imgId, catId):
        """Create gt, dt which are list of anns/dets.
        If params.useCats is true only anns/dets corresponding to tuple
        (imgId, catId) will be used.
        Else, all anns/dets in image are used and catId is not used.
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        return gt, dt

    def _sortDt(self, dt):
        # Sort detections in decreasing order of score.
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        return dt

    def _multihot(self, attributeIds):
        """Return a set of multi-class indices as multi-hot vectors.
        Arg:
            attributeIds (List[int]): list of multi-class integer labels,
                in the range of [0, numClasses-1]
        Returns:
            multihot (np.array): multi-hot vector, one for each of the
                elements in attribute_ids. (num_classes,)
        """
        numClasses = len(self.cocoGt.attrs)
        attid2continuous = {
            a: i for i, a in enumerate(self.cocoGt.attrs.keys())
        }

        multihot = np.zeros((numClasses), dtype=np.int32)
        for lab in attributeIds:
            multihot[attid2continuous[lab]] = 1

        return multihot

    def _f1score(self, true, predict):
        """
        compute f1_scores
        """
        if self.FPParams.f1Type == "binary_micro":
            return f1_score(true, predict, average="micro")
        if self.FPParams.f1Type == "binary_macro":
            return f1_score(true, predict, average="macro")

        true = true.reshape(1, -1)
        predict = predict.reshape(1, -1)
        # micro, macro
        return f1_score(true, predict, average=self.FPParams.f1Type)

    def computeF1(self, imgId, catId):
        """
        compute macro F1 score for attributes classification
        Returns:
            f1s (Numpy array): size (numDt, numGt).
                same size as computed IoUs
        """
        gt, dt = self._getGtDt(imgId, catId)
        if len(gt) == 0 and len(dt) == 0:
            return []

        dt = self._sortDt(dt)
        if len(dt) > self.params.maxDets[-1]:
            dt = dt[0:self.params.maxDets[-1]]

        # compute F1 between each dt and gt region
        f1s = np.zeros((len(dt), len(gt)))
        for dIdx, _dt in enumerate(dt):
            for gIdx, _gt in enumerate(gt):
                dAtts = self._multihot(_dt["attribute_ids"])
                gAtts = self._multihot(_gt["attribute_ids"])
                f1s[dIdx, gIdx] = self._f1score(gAtts, dAtts)
        return f1s

    def _expandDim(self, a):
        """expand array dimension
        Args:
            a (Numpy.Array): shape (ious, N) or (N)
        Return:
            a (Numpy.Array): shape (ious, 1, N) or (N, 1)
        """
        if len(a.shape) == 1:
            a = a[:, np.newaxis]
        return a[:, np.newaxis, :]

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        Returns:
            evalDdict (contains single image results)
        '''
        numF1Thrs = len(self.FPParams.f1Thrs)
        if catId not in self.FPParams.catsWithAttributes:
            evalDict = super(FPEval, self).evaluateImg(
                imgId, catId, aRng, maxDet
            )
            if evalDict is None:
                return None
            # change 'dtMatches','gtMatches', 'gtIgnore', 'dtIgnore' dim
            evalDict["dtMatches"] = self._expandDim(evalDict["dtMatches"])
            evalDict["gtMatches"] = self._expandDim(evalDict["gtMatches"])
            evalDict["gtIgnore"] = self._expandDim(evalDict["gtIgnore"])
            evalDict["dtIgnore"] = self._expandDim(evalDict["dtIgnore"])
            return evalDict

        gt, dt = self._getGtDt(imgId, catId)

        if len(gt) == 0 and len(dt) == 0:
            return None
        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dt = self._sortDt(dt)[:maxDet]

        # load computed ious
        ious = (
            self.ious[imgId, catId][:, gtind]
            if len(self.ious[imgId, catId]) > 0
            else self.ious[imgId, catId]
        )
        # load computed f1s
        f1s = (
            self.f1s[imgId, catId][:, gtind]
            if len(self.f1s[imgId, catId]) > 0
            else self.f1s[imgId, catId]
        )

        numIouThrs = len(self.params.iouThrs)
        numGt = len(gt)
        numDt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gtMatches = np.zeros((numIouThrs, numF1Thrs, numGt))
        dtMatches = np.zeros((numIouThrs, numF1Thrs, numDt))
        gtIgnore = np.array([g['_ignore'] for g in gt])
        dtIgnore = np.zeros((numIouThrs, numF1Thrs, numDt))

        for iouThrIdx, iouThr in enumerate(self.params.iouThrs):
            if len(ious) == 0:
                break

            for f1ThrIdx, f1Thr in enumerate(self.FPParams.f1Thrs):
                for dtIdx, _dt in enumerate(dt):
                    # match
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([iouThr, 1 - 1e-10])
                    m = -1
                    f1 = min([f1Thr, 1 - 1e-10])
                    for gtIdx, _ in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtMatches[iouThrIdx, f1ThrIdx, gtIdx] > 0:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIgnore[m] == 0 and \
                           gtIgnore[gtIdx] == 1:
                            break
                        # continue to next gt unless better match made
                        # (has to match 2 thresholds at the same time)
                        if ious[dtIdx, gtIdx] < iou or \
                           f1s[dtIdx, gtIdx] < f1:
                            continue
                        # if match successful and best so far, store appropriately  # noqa
                        iou = ious[dtIdx, gtIdx]
                        m = gtIdx
                        f1 = f1s[dtIdx, gtIdx]

                    # No match found for _dt, go to next _dt
                    if m == -1:
                        continue

                    # if gt to ignore for some reason update dtIgnore.
                    # Should not be used in evaluation.
                    dtIgnore[iouThrIdx, f1ThrIdx, dtIdx] = gtIgnore[m]
                    # _dt match found, update gtMatches, and dtMatches with "id"
                    dtMatches[iouThrIdx, f1ThrIdx, dtIdx] = gt[m]["id"]
                    gtMatches[iouThrIdx, f1ThrIdx, m] = _dt["id"]

        # set unmatched detections outside of area range to ignore
        dtIgnoreMask = [
            d["area"] < aRng[0]
            or d["area"] > aRng[1]
            for d in dt
        ]

        # 1 x numDt
        dtIgnoreMask = np.array(dtIgnoreMask).reshape((1, len(dt)))
        # numThrs x numDt
        dtIgnoreMask = np.repeat(dtIgnoreMask, numIouThrs, 0)
        # numThrs x 1 x numDt
        dtIgnoreMask = dtIgnoreMask.reshape((numIouThrs, 1, -1))
        # numThrs x numF1Thrs x numDt
        dtIgnoreMask = np.repeat(dtIgnoreMask, numF1Thrs, 1)

        # Based on dtIgnoreMask ignore any unmatched detection by updating dtIgnore  # noqa
        dtIgnore = np.logical_or(
            dtIgnore, np.logical_and(dtMatches == 0, dtIgnoreMask))
        # store results for given image and category
        return {
                'imageId':     imgId,
                'categoryId':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtMatches,
                'gtMatches':    gtMatches,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIgnore,
                'dtIgnore':     dtIgnore,
            }

    def accumulate(self, p=None, fpParams=None):
        '''
        Accumulate per image evaluation results,
        and store the result in self.evaluate
        '''
        print('Accumulating evaluation results...')

        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')

        # allows input customized parameters
        if p is None:
            p = self.params
        if fpParams is None:
            fpParams = self.FPParams

        p.catIds = p.catIds if p.useCats else [-1]
        numIous = len(p.iouThrs)
        numF1s = len(fpParams.f1Thrs)
        numRecalls = len(p.recThrs)
        numCats = len(p.catIds) if p.useCats else 1
        numAreaRngs = len(p.areaRng)
        numDets = len(p.maxDets)
        numImgs = len(p.imgIds)

        # -1 for absent categories
        precision = - np.ones(
            (numIous, numF1s, numRecalls, numCats, numAreaRngs, numDets)
        )
        recall = - np.ones(
            (numIous, numF1s, numCats, numAreaRngs, numDets)
        )
        scores = - np.ones(
            (numIous, numF1s, numRecalls, numCats, numAreaRngs, numDets)
        )

        # Initialize dtPointers
        dtPointers = {}
        for catIdx in range(numCats):
            dtPointers[catIdx] = {}
            for areaIdx in range(numAreaRngs):
                dtPointers[catIdx][areaIdx] = {}
                for dIdx in range(numDets):
                    dtPointers[catIdx][areaIdx][dIdx] = {}

        # retrieve E at each category, area range, and max number of detections
        # Per category evaluation
        for catIdx in range(numCats):
            if catIdx not in self.FPParams.catsWithAttributes:
                # if catId is a regular category without attributes
                # no need to loop over f1Thrs
                numF1s = 1

            Nk = catIdx * numAreaRngs * numImgs
            for areaIdx in range(numAreaRngs):
                Na = areaIdx * numImgs
                E = [
                    self.evalImgs[Nk + Na + imgIdx]
                    for imgIdx in range(numImgs)
                ]
                # Remove elements which are None
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                for dIdx, maxDet in enumerate(p.maxDets):
                    # Append all scores: shape (N,)
                    dtScores = np.concatenate(
                        [e['dtScores'][:maxDet] for e in E], axis=0)
                    dtIds = np.concatenate(
                        [e["dtIds"][:maxDet] for e in E], axis=0)

                    dtSortedIdx = np.argsort(-dtScores, kind='mergesort')
                    dtScores = dtScores[dtSortedIdx]
                    dtIds = dtIds[dtSortedIdx]

                    dtMatches = np.concatenate(
                        [e['dtMatches'][:, :, :maxDet] for e in E], axis=2
                    )[:, :, dtSortedIdx]   # numIou x numF1 x N
                    dtIgnore = np.concatenate(
                        [e["dtIgnore"][:, :, :maxDet] for e in E], axis=2
                    )[:, :, dtSortedIdx]   # numIou x numF1 x N
                    gtIgnore = np.concatenate([e["gtIgnore"] for e in E])
                    # num gt anns to consider
                    numGt = np.count_nonzero(gtIgnore == 0)

                    if numGt == 0:
                        continue

                    tps = np.logical_and(dtMatches, np.logical_not(dtIgnore))
                    fps = np.logical_and(
                        np.logical_not(dtMatches), np.logical_not(dtIgnore)
                    )

                    dtPointers[catIdx][areaIdx][dIdx] = {
                        "dtIds": dtIds,
                        "tps": tps,
                        "fps": fps,
                        "dtMatches": dtMatches,
                    }

                    tpSum = np.cumsum(tps, axis=2).astype(dtype=np.float)
                    fpSum = np.cumsum(fps, axis=2).astype(dtype=np.float)

                    for iouThrIdx in range(numIous):
                        for f1Idx in range(numF1s):
                            if numF1s == 1:
                                f1Idx = -1
                            tp = tpSum[iouThrIdx, f1Idx, :].reshape(-1)
                            fp = fpSum[iouThrIdx, f1Idx, :].reshape(-1)

                            numTp = len(tp)
                            rc = tp / numGt
                            if numTp:
                                recall[iouThrIdx, f1Idx, catIdx, areaIdx, dIdx] = rc[-1]  # noqa
                            # (numIous, numF1s, numCats, numAreaRngs, numDets)
                            else:
                                recall[iouThrIdx, f1Idx, catIdx, areaIdx, dIdx] = 0  # noqa

                            # np.spacing(1) ~= eps
                            pr = tp / (fp + tp + np.spacing(1))
                            pr = pr.tolist()

                            # Replace each precision value with the maximum precision
                            # value to the right of that recall level. This ensures
                            # that the calculated AP value will be less suspectable
                            # to small variations in the ranking.
                            for i in range(numTp - 1, 0, -1):
                                if pr[i] > pr[i - 1]:
                                    pr[i - 1] = pr[i]

                            recThrsInsertIdx = np.searchsorted(
                                rc, self.params.recThrs, side="left"
                            )

                            prAtRecall = [0.0] * numRecalls
                            _score = [0.0] * numRecalls

                            try:
                                for Idx, prIdx in enumerate(recThrsInsertIdx):
                                    prAtRecall[Idx] = pr[prIdx]
                                    _score[Idx] = dtScores[prIdx]
                            except:
                                pass

                            precision[iouThrIdx, f1Idx, :, catIdx, areaIdx, dIdx] = np.array(prAtRecall)  # noqa
                            # (numIous, numF1s, numRecalls, numCats, numAreaRngs, numDets)  # noqa
                            scores[iouThrIdx, f1Idx, :, catIdx, areaIdx, dIdx] = np.array(_score)  # noqa

                        # if numF1s == 1:
                        #     # repeat for all the dims
                        #     for a_f1Idx in range(1, len(fpParams.f1Thrs)):
                        #         precision[iouThrIdx, a_f1Idx, :, catIdx, areaIdx, dIdx] = np.array(prAtRecall)
                        #         scores[iouThrIdx, a_f1Idx, :, catIdx, areaIdx, dIdx] = np.array(_score)

            # reset numF1s
            numF1s = len(fpParams.f1Thrs)

        self.eval = {
            'params': p,
            'counts': [numIous, numF1s, numRecalls, numCats, numAreaRngs, numDets],  # noqa
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'dtPointers': dtPointers,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _summarize(
        self,
        summaryType,
        iouThr=None,
        f1Thr=None,
        f1=True,
        iou=True,
        areaRng='all',
        maxDets=100,
        catSuperIdx=None,
        catId=None,
    ):
        if catSuperIdx is not None and catId is not None:
            raise ValueError(
                "catId and catSuperIdx can not exist at the same time")
        if f1Thr is not None and catSuperIdx is not None and catSuperIdx == 1:
            raise ValueError(
                "catSuperIdx = 1 and f1Thr can not exist at the same time")

        p = self.params
        fpParams = self.FPParams

        aidx = [
            idx
            for idx, _areaRng in enumerate(p.areaRngLbl)
            if _areaRng == areaRng
        ]

        midx = [
            idx
            for idx, _mDet in enumerate(p.maxDets)
            if _mDet == maxDets
        ]

        if summaryType == 'ap':
            # dimension of precision:
            # (numIous, numF1s, numRecalls, numCats, numAreaRngs, numDets)
            s = self.eval['precision']

            if iou and iouThr is not None:
                tIdx = np.where(iouThr == p.iouThrs)[0]
                s = s[tIdx, :, :, :, :, :]
            elif iou and iouThr is None:
                s = s[:-1, :, :, :, :, :]
            elif not iou:
                s = s[[-1], :, :, :, :, :]

            if f1 and f1Thr is not None:
                # filter out cats that not have attributes
                fIdx = np.where(f1Thr == fpParams.f1Thrs)[0]

                arrayList = []
                for c in range(len(self.cats)):
                    if catId is not None and c != catId:
                        continue
                    if catSuperIdx is not None and \
                       c not in self.superCats[catSuperIdx]:
                        continue

                    if c in fpParams.catsWithAttributes:
                        _s = s[:, fIdx, :, c, :, :]
                    else:
                        _s = s[:, [-1], :, c, :, :]

                    _s = _s[:, :, :, np.newaxis, :, :]
                    arrayList.append(_s)

                s = np.concatenate(arrayList, 3)  # (1, 10, 101, 46, 4, 3)

            elif f1 and f1Thr is None:
                arrayList = []
                for c in range(len(self.cats)):
                    if catId is not None and c != catId:
                        continue
                    if catSuperIdx is not None and \
                       c not in self.superCats[catSuperIdx]:
                        continue

                    if c in fpParams.catsWithAttributes:
                        for fIdx in range(len(fpParams.f1Thrs[:-1])):
                            _s = s[:, fIdx, :, c, :, :]
                            _s = _s[:, np.newaxis, :, np.newaxis, :, :]
                            arrayList.append(_s)
                    else:
                        _s = s[:, [-1], :, :, :, :]
                        _s = _s[:, :, :, [c], :, :]
                        arrayList.append(_s)

                # (10, 1, 101, 208, 4, 3) reshape f1_axis to cat axis
                # 208 = numF1s * numCats_w_atts + numCats_wo_atts
                s = np.concatenate(arrayList, 3)

            elif not f1:
                s = s[:, [-1], :, :, :, :]
                if catSuperIdx is not None:
                    catIds = self.superCats[catSuperIdx]
                    s = s[:, :, :, catIds, :, :]
                elif catId is not None:
                    s = s[:, :, :, [catId], :, :]
                else:
                    s = s[:, :, :, :, :, :]

            s = s[:, :, :, :, aidx, midx]
        else:
            # dimension of recall:
            # (numIous, numF1s, numCats, numAreaRngs, numDets)
            s = self.eval['recall']

            if iou and iouThr is not None:
                tIdx = np.where(iouThr == p.iouThrs)[0]
                s = s[tIdx, :, :, :, :]
            elif iou and iouThr is None:
                s = s[:-1, :, :, :, :]
            elif not iou:
                s = s[[-1], :, :, :, :]

            if f1 and f1Thr is not None:
                fIdx = np.where(f1Thr == fpParams.f1Thrs)[0]
                arrayList = []
                for c in range(len(self.cats)):
                    if catId is not None and c != catId:
                        continue
                    if catSuperIdx is not None and \
                       c not in self.superCats[catSuperIdx]:
                        continue

                    if c in fpParams.catsWithAttributes:
                        _s = s[:, fIdx, c, :, :]
                    else:
                        _s = s[:, [-1], c, :, :]

                    _s = _s[:, :, np.newaxis, :, :]
                    arrayList.append(_s)
                s = np.concatenate(arrayList, 2)  # (1, 10, 46, 4, 3)

            elif f1 and f1Thr is None:
                arrayList = []
                for c in range(len(self.cats)):
                    if catId is not None and c != catId:
                        continue
                    if catSuperIdx is not None and \
                       c not in self.superCats[catSuperIdx]:
                        continue

                    if c in fpParams.catsWithAttributes:
                        for fIdx in range(len(fpParams.f1Thrs[:-1])):
                            _s = s[:, fIdx, c, :, :]
                            _s = _s[:, np.newaxis, np.newaxis, :, :]
                            arrayList.append(_s)
                    else:
                        _s = s[:, -1, c, :, :]
                        _s = _s[:, np.newaxis, np.newaxis, :, :]
                        arrayList.append(_s)

                s = np.concatenate(arrayList, 2)
            elif not f1:
                s = s[:, [-1], :, :, :]
                if catSuperIdx is not None:
                    catIds = self.superCats[catSuperIdx]
                    s = s[:, :, catIds, :, :]
                elif catId is not None:
                    catIds = [catId]
                    s = s[:, :, catIds, :, :]
                else:
                    s = s[:, :, :, :, :]

            s = s[:, :, :, aidx, midx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s

    def run(self):
        """Wrapper function which calculates the results."""
        self.evaluate()
        self.accumulate()
        self.summarize()

    def summarize(self):
        self.results = self._summarize_with_cat()
        self.resultDet = self._summarize_with_cat(f1=False)
        self.resultF1 = self._summarize_with_cat(iou=False)

    def _summarize_with_cat(self, f1=True, iou=True, catIdx=None, catSuperIdx=None):
        '''
        Compute and display summary metrics for evaluation results.
        This functin can *only* be applied on the default parameter setting
        '''
        if not self.eval:
            raise Exception('Please run accumulate() first')
        if not f1 and not iou:
            raise ValueError("At least one of iou and f1 constraint should be true")
        if catIdx is not None and catSuperIdx is not None:
            raise ValueError(
                "catId and catSuperIdx can not exist at the same time")

        results = {}
        kwargs = {
            "f1": f1,
            "iou": iou,
        }

        if catSuperIdx is None and catIdx is None:
            for idx, super_cls in enumerate(self.FPParams.catSuperCls):
                if idx == 1 and f1:
                    continue
                results["AP" + super_cls] = self._summarize(
                    'ap', catSuperIdx=idx, **kwargs)

        kwargs["catSuperIdx"] = catSuperIdx
        kwargs["catId"] = catIdx
        results["AP"] = self._summarize('ap', **kwargs)
        if iou:
            results["AP_IOU50"] = self._summarize('ap', iouThr=.5, **kwargs)
            results["AP_IOU75"] = self._summarize('ap', iouThr=.75, **kwargs)

        if f1:
            for f1Thr in [0.5, 0.75]:
                key = "AP_F1{:d}".format(int(f1Thr * 100))
                results[key] = self._summarize("ap", f1Thr=f1Thr, **kwargs)

        if f1 and iou:
            for iouThr in [0.5, 0.75]:
                for f1Thr in [0.5, 0.75]:
                    key = "AP_IOU{:d}_F1{:d}".format(
                        int(iouThr * 100), int(f1Thr * 100))
                    results[key] = self._summarize(
                        "ap", iouThr=iouThr, f1Thr=f1Thr, **kwargs)

        results["APs"] = self._summarize('ap', areaRng='small', **kwargs)
        results["APm"] = self._summarize('ap', areaRng='medium', **kwargs)
        results["APl"] = self._summarize('ap', areaRng='large', **kwargs)

        for maxDet in self.params.maxDets:
            key = "AR@{}".format(maxDet)
            results[key] = self._summarize("ar", maxDets=maxDet, **kwargs)

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@100".format(area_rng[0])
            results[key] = self._summarize('ar', areaRng=area_rng, **kwargs)
        return results

    def summarize_class(self, perSuperClass=True, perCls=True):
        if not self.eval:
            raise Exception('Please run accumulate() first')
        result = {}
        if perSuperClass:
            result.update(self._per_supercls_summarize())
        if perCls:
            result.update(self._per_cls_summarize())
        self.results_per_class = result

    def _per_cls_summarize(self):
        """compute result per class"""
        per_class_result = {}
        for catId in range(len(self.cats)):
            resultDet = self._summarize_with_cat(f1=False, catIdx=catId)

            if catId in self.FPParams.catsWithAttributes:
                results = self._summarize_with_cat(catIdx=catId)
                resultF1 = self._summarize_with_cat(iou=False, catIdx=catId)

                per_class_result[self.cats[catId]["name"]] = {
                    "iou_f1": results, "f1": resultF1, "iou": resultDet,
                }
            else:
                per_class_result[self.cats[catId]["name"]] = {"iou": resultDet}

        return per_class_result

    def _per_supercls_summarize(self):
        """compute result per superclass"""

        per_super_class_result = {}
        superCats = self.FPParams.catSuperClsName

        for superCatId, superCat in enumerate(superCats):
            superCatKey = "supercls-" + superCat

            resultDet = self._summarize_with_cat(
                f1=False, catSuperIdx=superCatId)
            per_super_class_result[superCatKey] = {"iou": resultDet}
            if superCatId != 1:
                results = self._summarize_with_cat(catSuperIdx=superCatId)
                resultF1 = self._summarize_with_cat(
                    iou=False, catSuperIdx=superCatId)

                per_super_class_result[superCatKey]["f1"] = resultF1
                per_super_class_result[superCatKey]["iou + f1"] = results

        return per_super_class_result

    def print(self, f1=True, iou=True):
        if len(self.results) == 0:
            raise Exception('Please run run() first')
        if not f1 and not iou:
            raise ValueError("At least one of iou and f1 constraint should be true")

        if f1 and iou:
            result_dict = self.results
            header = "results with both IoU and F1 thresholds"
            iou = "{:0.2f}:{:0.2f}".format(
                self.params.iouThrs[0], self.params.iouThrs[-2]
            )
            f1 = "{:0.2f}:{:0.2f}".format(
                self.FPParams.f1Thrs[0], self.FPParams.f1Thrs[-2]
            )
        elif f1:
            result_dict = self.resultF1
            header = "results with F1 thresholds (iou_threshold = -1.0)"
            iou = "none"
            f1 = "{:0.2f}:{:0.2f}".format(
                self.FPParams.f1Thrs[0], self.FPParams.f1Thrs[-2]
            )
        else:
            result_dict = self.resultDet
            header = "results with IoU thresholds (f1_threshold = -1.0)"
            iou = "{:0.2f}:{:0.2f}".format(
                self.params.iouThrs[0], self.params.iouThrs[-2]
            )
            f1 = "none"
        self._print_results(result_dict, header, f1, iou)

    def print_class_result(self, className, f1=True, iou=True):
        if len(self.results_per_class) == 0:
            raise Exception('Please run summarize_class() first')
        if className not in self.results_per_class:
            raise ValueError("{} is not a valid class name".format(className))
        if not f1 and not iou:
            raise ValueError("At least one of iou and f1 constraint should be true")

        if f1 and iou:
            result_dict = self.results_per_class[className]["iou_f1"]
            header = "{} results with both IoU and F1 thresholds".format(className)
            iou = "{:0.2f}:{:0.2f}".format(
                self.params.iouThrs[0], self.params.iouThrs[-2]
            )
            f1 = "{:0.2f}:{:0.2f}".format(
                self.FPParams.f1Thrs[0], self.FPParams.f1Thrs[-2]
            )
        elif f1:
            result_dict = self.results_per_class[className]["f1"]
            header = "{} results with F1 thresholds (iou_threshold = -1.0)".format(className)
            iou = "none"
            f1 = "{:0.2f}:{:0.2f}".format(
                self.FPParams.f1Thrs[0], self.FPParams.f1Thrs[-2]
            )
        else:
            result_dict = self.results_per_class[className]["iou"]
            header = "{} results with IoU thresholds (f1_threshold = -1.0)".format(className)
            iou = "{:0.2f}:{:0.2f}".format(
                self.params.iouThrs[0], self.params.iouThrs[-2]
            )
            f1 = "none"
        self._print_results(result_dict, header, f1, iou, False)

    def _print_results(
        self,
        result_dict,
        header,
        default_f1,
        default_iou,
        displaySuperCls=True
    ):
        if displaySuperCls:
            template = " {:<18} {} @[ IoU={:<9} | F1={:<9} |area={:>3s} | maxDets={:>3d} | superCat={:>9s}] = {:0.3f}"  # noqa
        else:
            template = " {:<18} {} @[ IoU={:<9} | F1={:<9} |area={:>3s} | maxDets={:>3d} ] = {:0.3f}"  # noqa

        cat2name = {aname: name for aname, name in zip(
            self.FPParams.catSuperCls, self.FPParams.catSuperClsName)}
        print("=" * 80)
        print(header)
        print("=" * 80)
        for key, value in result_dict.items():
            if "@" in key:
                maxDets = int(key.split("@")[-1])
            else:
                maxDets = self.params.maxDets[2]
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            else:
                title = "Average Recall"
                _type = "(AR)"

            if "F1" in key and "IOU" in key:
                # "AP_IOU50_F150"
                names = key.split("_")
                iouThr = (float(names[1][3:]) / 100)
                iou = "{:0.2f}".format(iouThr)
                f1Thr = (float(names[2][2:]) / 100)
                f1 = "{:0.2f}".format(f1Thr)
            elif "F1" in key:
                # AP_F150
                names = key.split("_")
                f1Thr = (float(names[1][2:]) / 100)
                f1 = "{:0.2f}".format(f1Thr)
                iou = default_iou
            elif "IOU" in key:
                # AP_IOU50
                names = key.split("_")
                iouThr = (float(names[1][3:]) / 100)
                iou = "{:0.2f}".format(iouThr)
                f1 = default_f1
            else:
                f1 = default_f1
                iou = default_iou

            if len(key) > 2 and key[2] in self.FPParams.catSuperCls:
                cat_group_name = cat2name[key[2]]
            else:
                cat_group_name = "all"

            if len(key) > 2 and key[2] in ["s", "m", "l"]:
                area_rng = key[2]
            else:
                area_rng = "all"

            if displaySuperCls:
                print(template.format(
                    title, _type, iou, f1, area_rng,
                    maxDets, cat_group_name, value
                ))
            else:
                print(template.format(
                    title, _type, iou, f1, area_rng, maxDets, value
                ))
        print("=" * 80)


class FPParams:
    '''
    Params specific for Fashionpedia eval
    '''
    def __init__(self):
        self.catsWithAttributes = [
            10, 7, 0, 6, 5, 4, 9, 31, 33, 3, 12, 11, 1, 2, 8, 32, 29, 28,
        ]
        self.catsWithOutAttributes = [
            c for c in range(46) if c not in self.catsWithAttributes
        ]

        # category super class
        # o: Outerwear
        # p: Garment Parts
        # a: Accessory
        self.catSuperCls = ["o", "a", "p"]
        self.catSuperClsName = ["outerwear", "accessory", "part"]
        # hard-coded super category ids
        self.catSuperClsIds = [
            np.arange(13),
            np.arange(13, 27),
            np.arange(27, 46)
        ]

        # attributes F1 score threshold
        self.f1Thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.f1Thrs = np.append(self.f1Thrs, -1.0)

        # f1 type: binary_micro, binary_macro, micro, macro
        self.f1Type = "binary_macro"
