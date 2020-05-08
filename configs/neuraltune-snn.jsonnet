local project_root = std.extVar('project_root');
local rep_hidden = 200;
local reg_hidden = 5;
local pruned_ms = ['driver.jvm.non-heap.committed.avg_period', 'worker_1.Disk_Write_KB/s.dm-0', 'executor.jvmGCTime.avg', 'driver.jvm.heap.max.avg', 'worker_2.Processes.Blocked', 'driver.jvm.pools.Compressed-Class-Space.used.avg_increase', 'latency'];
local scaler_p = 'pruned_metrics3.pkl';
{
  random_seed: 123,
  dataset_reader: {
    type:
      'neuraltune-reader',
    type_flag: 'train',
    pruned_metrics: pruned_ms,
    scaler_path: scaler_p,

  },
  validation_dataset_reader: {
    type:
      'neuraltune-reader',
    type_flag: 'dev',
    pruned_metrics: pruned_ms,
    scaler_path: scaler_p,
  },
  train_data_path: project_root + '/.data',
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
      input_dim: 12 + std.length(pruned_ms),
      hidden_dims: rep_hidden,
      num_layers: 3,
      dropout: 0.1,
      activations: 'tanh',
    },
    regression_network: {
      input_dim: 12 + rep_hidden,
      hidden_dims: reg_hidden,
      num_layers: 2,
      activations: 'relu',
    },
    scaler_path: scaler_p,

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
        type: 'checkpoint',
        checkpointer: {
          num_serialized_models_to_keep: 1,
        },
      },

      {
        type: 'update_learning_rate',
        learning_rate_scheduler: {
          type: 'reduce_on_plateau',
          factor: 0.8,
          mode: 'min',
          patience: 1,
        },
      },
    ],
    optimizer: {
      type: 'adam',
      lr: 0.00005,
      weight_decay: 0.01,
    },
    cuda_device: -1,
    num_epochs: 50,
    shuffle: true,
  },
}
