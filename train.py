# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Training executable for detection models.

This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""
"""导入需要的包"""
import functools
import json #用 Python 语言来编码和解码 JSON 对象
import os #文件和目录的操作
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""调用object_detection相关的包"""
from object_detection.legacy import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util

#添加该语句即可通过tensorboard来看出训练过程中loss accuracy的变化
tf.logging.set_verbosity(tf.logging.INFO)

"""定义了tf.app.flags，用于支持接受命令行传递参数"""
flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')#GPU还是CPU，如果GPU内存不够大，务必使用CPU clones.false改为True
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.') #train_dir目录用于保存训练的模型和日志文件

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')#用于指定PipelineConfig配置文件的全路径
"""以下三个参数都被包含在pipeline_config_path中"""
flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS

"""定义主函数"""
def main(_):
  assert FLAGS.train_dir, '`train_dir` is missing.'
  if FLAGS.task == 0: tf.gfile.MakeDirs(FLAGS.train_dir) # tf.gfile模块创建一个目录
  if FLAGS.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)#读取pipeline_config_path配置文件，返回一个dict，保存配置文件中`model`, `train_config`,
    #`train_input_config`, `eval_config`, `eval_input_config`信息
    if FLAGS.task == 0:
      tf.gfile.Copy(FLAGS.pipeline_config_path,
                    os.path.join(FLAGS.train_dir, 'pipeline.config'),
                    overwrite=True) #把pipeline_config_path配置文件复制到train_dir目录下，命名为pipeline.config
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        train_config_path=FLAGS.train_config_path,
        train_input_config_path=FLAGS.input_config_path)#读取model_config_path、train_config_path、train_input_config_path的路径
    if FLAGS.task == 0:
      for name, config in [('model.config', FLAGS.model_config_path),
                           ('train.config', FLAGS.train_config_path),
                           ('input.config', FLAGS.input_config_path)]:
        tf.gfile.Copy(config, os.path.join(FLAGS.train_dir, name),
                      overwrite=True)

  model_config = configs['model']
  train_config = configs['train_config']
  input_config = configs['train_input_config']

  """"
  以下这行代码为核心代码，通过传入部分所需要的参数并且 “重新定义” 函数名称。这样简化函数,更少更灵活的函数参数调用。 
  通过functools.partial函数对model_builder.build函数赋予默认值，该目录下有一个model_builder模块，包含了生成网络模型的代码，
  包含ssd,fast_rcnn等众多模型代码，部分代码如下所示
  def build(model_config, is_training):
      if not isinstance(model_config, model_pb2.DetectionModel):
          raise ValueError('model_config not of type model_pb2.DetectionModel.')
      # 获取配置中的模型种类
      meta_architecture = model_config.WhichOneof('model')
      # 进行具体加载
      if meta_architecture == 'ssd':
          return _build_ssd_model(model_config.ssd, is_training)
      if meta_architecture == 'faster_rcnn':
          return _build_faster_rcnn_model(model_config.faster_rcnn, is_training)
      raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
      以'faster_rcnn模型为例子，进入_build_faster_rcnn_model（仍在model_builder.py文件中），该类中定义了fast_rcnn所有的参数
      之后说明每一个子模型的构建，比如image_resizer_builder的构建
      """

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)
  #第二阶段中的参数配置
  def get_next(config):
    return dataset_builder.make_initializable_iterator(
        dataset_builder.build(config)).get_next()

  create_input_dict_fn = functools.partial(get_next, input_config)
  #python解码JSON对象
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  graph_rewriter_fn = None
  if 'graph_rewriter_config' in configs:
    graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=True)

  trainer.train(
      create_input_dict_fn,
      model_fn,
      train_config,
      master,
      task,
      FLAGS.num_clones,
      worker_replicas,
      FLAGS.clone_on_cpu,
      ps_tasks,
      worker_job_name,
      is_chief,
      FLAGS.train_dir,
      graph_hook_fn=graph_rewriter_fn)


if __name__ == '__main__':
  tf.app.run()


#CUDA_VISIBLE_DEVICES=0,1
# python train.py --logtostderr --pipeline_config_path=D:/ship/ship_name/test/data/faster_rcnn_inception_v2_coco.config --train_dir=D:/ship/ship_name/test/train_inc --num_clones=2 --ps_tasks=1
# python train.py --logtostderr --pipeline_config_path=D:/ship/ship_name/test/gao_data/faster_rcnn_resnet101_coco.config --train_dir=D:/ship/ship_name/test/gao_data/Gao1705_resnet_train --num_clones=2 --ps_tasks=1
# python train.py --logtostderr --pipeline_config_path=D:/ship/ship_name/test/gao_data/faster_rcnn_resnet101_coco.config --train_dir=D:/ship/ship_name/test/gao_data/Gao1705_resnet_train/train_temp_test --num_clones=2 --ps_tasks=1

# python train.py --logtostderr --pipeline_config_path=D:/ship/ship_name/test/gao_data/faster_rcnn_resnet101_coco.config --train_dir=D:/ship/ship_name/test/gao_data/train_resnet
