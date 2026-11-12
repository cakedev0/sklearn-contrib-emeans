py.extension_module(
  '_kernels',
  '_kernels.pyx',
  dependencies: [openmp_dep],
  install: true,
  subdir: 'sklearn_emeans',
)

py.extension_module(
  '_kmeans_plusplus_kernels',
  '_kmeans_plusplus_kernels.pyx',
  dependencies: [openmp_dep],
  install: true,
  subdir: 'sklearn_emeans',
)

py.install_sources(
  ['__init__.py', '_emeans.py', '_kmeans_parallel.py', '_kmeans_plusplus.py'],
  subdir: 'sklearn_emeans',
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_common.py',
    'tests/test_emeans.py',
    'tests/test_kmeans_plusplus.py',
  ],
  subdir: 'sklearn_emeans/tests',
)
