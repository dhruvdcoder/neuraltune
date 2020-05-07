local project_root = std.extVar('project_root');
{
  dataset_reader: {
    type:
      'neuraltune-reader',
    type_flag: 'train',
    pruned_metrics: ['executor.jvm.non-heap.committed.avg_increase', 'executor.resultSerializationTime.avg', 'worker_1.Paging_and_Virtual_Memory.pgpgout', 'worker_1.Disk_Write_KB/s.sdi', 'worker_1.Disk_Read_KB/s.dm-0', 'executor.jvm.non-heap.committed.avg_period', 'executor.jvm.pools.PS-Old-Gen.committed.avg', 'latency'],
    scaler_path: 'pruned_metrics1.pkl',
  },
  validation_dataset_reader: {
    type:
      'neuraltune-reader',
    type_flag: 'dev',
    pruned_metrics: ['executor.jvm.non-heap.committed.avg_increase', 'executor.resultSerializationTime.avg', 'worker_1.Paging_and_Virtual_Memory.pgpgout', 'worker_1.Disk_Write_KB/s.sdi', 'worker_1.Disk_Read_KB/s.dm-0', 'executor.jvm.non-heap.committed.avg_period', 'executor.jvm.pools.PS-Old-Gen.committed.avg', 'latency'],
    scaler_path: 'pruned_metrics1.pkl',
  },
  train_data_path: project_root + '/.data',
  validation_data_path: project_root + '/.data',
  datasets_for_vocab_creation: [],
  iterator: {
    type: 'basic',
    batch_size: 32,
    cache_instances: false,
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
      input_dim: 23,
      hidden_dims: 5,
      num_layers: 1,
      activations: 'tanh',
    },
    regression_network: {
      input_dim: 17,
      hidden_dims: 5,
      num_layers: 1,
      activations: 'tanh',
    },
    scaler_path: 'pruned_metrics1.pkl',

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
        patience: 15,
        validation_metric: '-loss',
      },
    ],
    optimizer: {
      type: 'adam',
      lr: 0.005,
    },
    cuda_device: -1,
    num_epochs: 15,
    shuffle: true,
  },
}
