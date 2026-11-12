py.extension_module(
  '_kernels',
  '_kernels.pyx',
  dependencies: [openmp_dep],
  install: true,
  subdir: 'sklearn_emeans',
)

py.install_sources(
  ['__init__.py', '_emeans.py', '_kmeans_parallel.py'],
  subdir: 'sklearn_emeans',
)

py.install_sources(
  ['tests/__init__.py', 'tests/test_emeans.py', 'tests/test_common.py'],
  subdir: 'sklearn_emeans/tests',
)
