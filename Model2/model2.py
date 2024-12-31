import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection import model_lib_v2
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util


# Path to the Faster R-CNN config and checkpoint
pipeline_config = 'PATH_TO_FASTER_RCNN_PIPELINE_CONFIG'
checkpoint_path = 'PATH_TO_FASTER_RCNN_CHECKPOINT'

# Load model configuration
pipeline_config = 'PATH_TO_FASTER_RCNN_PIPELINE_CONFIG'
configs = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(pipeline_config, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, configs)

# Build the model
model_config = configs.model
model = model_util.create_model(model_config, is_training=False)

# Load pre-trained weights
ckpt = tf.compat.v2.train.Checkpoint(model=model)
ckpt.restore(checkpoint_path).expect_partial()
