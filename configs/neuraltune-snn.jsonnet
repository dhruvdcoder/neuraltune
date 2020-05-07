local project_root = std.extVar('project_root');
local rep_hidden = 50;
local reg_hidden = 5;
{
  dataset_reader: {
    type:
      'neuraltune-static-reader',
  },
  validation_dataset_reader: {
    type:
      'neuraltune-reader',
    type_flag: 'dev',
  },
  train_data_path: project_root + '/.data/train_worker_2.Processes.Blocked__executor.runTime.avg__worker_2.Disk_%Busy.sdi3__latency_10_samples.pkl',
  validation_data_path: project_root + '/.data',
  datasets_for_vocab_creation: [],
  iterator: {
    type: 'basic',
    batch_size: 32,
    cache_instances: true,
  },
  validation_iterator: {
    type: 'basic',
    batch_size: 32,
    cache_instances: false,
  },
  model: {
    type: 'simple-nn',
    num_samples: 5,
    representation_network: {
      input_dim: 16,
      hidden_dims: rep_hidden,
      num_layers: 3,
      dropout: 0.1,
      activations: 'relu',
    },
    regression_network: {
      input_dim: 12 + rep_hidden,
      hidden_dims: 50,
      num_layers: 2,
      activations: 'relu',
    },
  },
  trainer: {
    type: 'callback',
    local common_debug = false,
    local common_freq = 2,
    callbacks: [
      {
        type: 'validate',
      },
      {
        type: 'track_metrics',
        patience: 5,
        validation_metric: '-loss',
      },
      {
        type: 'update_learning_rate',
        learning_rate_scheduler: {
          type: 'reduce_on_plateau',
          factor: 0.5,
          mode: 'min',
          patience: 0,
        },
      },
    ],
    optimizer: {
      type: 'adam',
      lr: 0.0005,
      weight_decay: 0.01,
    },
    cuda_device: -1,
    num_epochs: 30,
    shuffle: true,
  },
}
