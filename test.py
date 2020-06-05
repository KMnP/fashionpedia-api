from fashionpedia.fp import Fashionpedia
from fashionpedia.fp_eval import FPEval

annType = 'segm'
anno_file = "./data/sample.json"
res_file = "./data/fake_results.json"

fpGt = Fashionpedia(anno_file)
fpDt = fpGt.loadRes(res_file)
imgIds = sorted(fpGt.getImgIds())

# run evaluation
fp_eval = FPEval(fpGt, fpDt, annType)
fp_eval.params.imgIds = imgIds
fp_eval.run()
fp_eval.print()
