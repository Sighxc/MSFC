from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import cv2
def main():
    #将fpn的输出p2~p5输入到faster rcnn中
    register_coco_instances("val2017_cat", {}, "C:/Users/Sigh/PycharmProjects/val2017_cat_ann/val2017_cat_ann.json",
                            "C:/Users/Sigh/PycharmProjects/val2017_cat")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("val2017_cat",)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("val2017_cat")
    dataset_dicts = DatasetCatalog.get("val2017_cat")
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                        metadata=metadata,
                        scale=0.8,
                        instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    evaluator = COCOEvaluator("val2017_cat", cfg, False, output_dir="C:/Users/Sigh/PycharmProjects/output/")
    val_loader = build_detection_test_loader(cfg, "val2017_cat")
    inference_on_dataset(predictor.model, val_loader, evaluator)

if __name__=='__main__': #不加这句就会报错
    main()