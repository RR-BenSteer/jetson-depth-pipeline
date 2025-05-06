project('DepthPipe', 'cpp', 'cuda',
  default_options: ['cpp_std=c++17', 'buildtype=debug']
)

thirdparty_build = custom_target('thirdparty_build',
  output: 'thirdparty_build_stamp',
  command: ['bash', '-c', 'cd ' + meson.project_source_root() + ' && ./scripts/build_thirdparty.sh'],
  build_by_default: true
)

tensorrt_lib = custom_target('tensorrt_lib',
  input: thirdparty_build,
  output: 'libtensorrt_cpp_api.so',
  command: ['ln', '-sf', meson.project_source_root() / 'Thirdparty/tensorrt-cpp-api/lib/libtensorrt_cpp_api.so', '@OUTPUT@'],
  build_by_default: true
)

cudasift_lib = custom_target('cudasift_lib',
  input: thirdparty_build,
  output: 'libcudasift.so',
  command: ['ln', '-sf', meson.project_source_root() / 'Thirdparty/CudaSift/lib/libcudasift.so', '@OUTPUT@'],
  build_by_default: true
)

deps = [
  dependency('cuda', required: true),
  dependency('opencv4', required: true),
  dependency('eigen3', required: true),
  dependency('fmt', required: true),
  dependency('threads'),
]

message('Build type: ' + get_option('buildtype'))

arch = host_machine.cpu_family()
if arch == 'aarch64' or arch == 'arm'
  message('Building for an ARM architecture: ' + arch)
endif

inc_dirs = include_directories([
  '.',
  'include',
  'Thirdparty/CudaSift',
  'Thirdparty/tensorrt-cpp-api/include',
  'Thirdparty/tensorrt-cpp-api/include/interfaces',
  'Thirdparty/tensorrt-cpp-api/src',
])

# CUDA flags
nvcc_args = ['-O2', '-D_FORCE_INLINES', '-DVERBOSE_NOT', '--compiler-options', '-O2']

link_args = [ '-lnvinfer', '-lnvonnxparser' ]

if arch == 'aarch64'
  cuda_arch = ['-gencode=arch=compute_72,code=sm_72']
else
  cuda_arch = []
endif

cpp_args = ['-Wall', '-O3', '-g', '-march=native']

src_files = files([
  'src/DepthAnything.cc',
  'src/DepthPipe.cc',
  'src/model.cc',
  'src/estimator.cu',
])

libdepthpipe = static_library(
  'DepthPipe',
  sources: src_files,
  include_directories: inc_dirs,
  cpp_args: cpp_args,
  cuda_args: nvcc_args + cuda_arch,
  dependencies: deps,
  install: false,
  link_args : link_args,
  link_with: [cudasift_lib, tensorrt_lib],
)

# Export the dependencies for this subproject
depthpipe_dep = declare_dependency(
  include_directories: inc_dirs,
  dependencies: deps,
  link_args : link_args,
  link_with: libdepthpipe,
)

# examples_dir = 'Examples'
# executable_names = ['lizard_test', 'flsea_test', 'test_engine']

# foreach exe : executable_names
#   exe_src = join_paths(examples_dir, exe + '.cc')
#   executable(exe,
#     exe_src,
#     include_directories: inc_dirs,
#     dependencies: deps,
#     link_with: libdepthpipe,
#     install: false
#   )
# endforeach