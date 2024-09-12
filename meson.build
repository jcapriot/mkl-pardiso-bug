project(
  'pydiso',
  'cpp',
  'v0.0.1',
  license: 'MIT',
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