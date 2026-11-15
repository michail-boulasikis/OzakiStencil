add_cpu_lib(stencil-spec stencil_spec.cpp)
target_link_libraries(stencil-spec PUBLIC quadmath)

add_library(stencil-registry STATIC stencil_registry.cpp)
target_link_libraries(stencil-registry PUBLIC ozaki_common stencil-spec quadmath)
