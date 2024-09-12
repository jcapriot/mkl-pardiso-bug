project(
  'pydiso',
  'c', 'cython',
  'v0.0.1',
  license: 'MIT',
  meson_version: '>= 1.4.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
  ],
)

cpp = meson.get_compiler('cpp')

mkl_dep = dependency('mkl-sdl')

executable(
    'test_solve',
    [
        'test_solve.cpp',
    ],
    dependencies : mkl_dep,
)