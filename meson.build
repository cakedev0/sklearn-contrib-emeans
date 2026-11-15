project(
  'sklearn-contrib-emeans',
  'c', 'cython',
  version: '0.3.2',
  license: 'BSD-3-Clause',
  meson_version: '>= 1.1.0',
  default_options: ['buildtype=release', 'c_std=c11'],
)

py = import('python').find_installation(pure: false)
cc = meson.get_compiler('c')

openmp_dep = dependency('OpenMP', language: 'c', required: false)
if not openmp_dep.found()
  # On macOS with Apple Clang, OpenMP is usually enabled through the CPPFLAGS
  # and LDFLAGS environment variables, which this dependency doesn't detect.
  warning('OpenMP not detected: unless it is enabled by compiler flags, ' +
          'EMeans will run sequentially.')
endif

subdir('sklearn_emeans')
