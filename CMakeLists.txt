
#execute_process (
#    COMMAND bash -c "source /ccs/home/jiquntu/quda-build/develop/build-mpi/load_module.sh"
#)

find_package(MPI)
find_package(CUDA)

set( QUDA_DIR "/home/jiquntu/quda-build/develop/build-mpi/" )

include_directories( . "${QUDA_DIR}/include" ${MPI_INCLUDE_PATH} )
link_directories( . "${QUDA_DIR}/lib")

set( STUPID_FLAGS "-std=gnu++11 -DBUILD_CPS_INTERFACE -DBUILD_MILC_INTERFACE -DBUILD_QDP_INTERFACE -DGPU_CLOVER_DIRAC -DGPU_DOMAIN_WALL_DIRAC -DGPU_GAUGE_TOOLS -DGPU_STAGGERED_DIRAC -DGPU_TWISTED_CLOVER_DIRAC -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC -DMAX_MULTI_BLAS_N=4 -DMPI_COMMS -DMULTI_GPU -D__COMPUTE_CAPABILITY__=700" )

set( CMAKE_CXX_FLAGS ${STUPID_FLAGS} )
set( CMAKE_C_FLAGS ${STUPID_FLAGS} )

#build a common library for all test utilities
set(QUDA_TEST_COMMON gtest-all.cc test_util.cpp misc.cpp face_gauge.cpp)
cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})

set(TEST_LIBS quda quda_test )

MACRO (QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  IF(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  ENDIF()
ENDMACRO()
#define tests

cuda_add_executable(dslash_test dslash_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
target_link_libraries(dslash_test ${TEST_LIBS} ${MPI_LIBRARIES} ${CUDA_LIBRARIES} cuda )
#target_link_libraries(dslash_test ${TEST_LIBS} ${CUDA_LIBRARIES} cuda)
QUDA_CHECKBUILDTEST(dslash_test QUDA_BUILD_ALL_TESTS)

cuda_add_executable(invert_test invert_test.cpp wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
target_link_libraries(invert_test ${TEST_LIBS} ${MPI_LIBRARIES} ${CUDA_LIBRARIES} cuda )
#target_link_libraries(invert_test ${TEST_LIBS} ${CUDA_LIBRARIES} cuda)
QUDA_CHECKBUILDTEST(invert_test QUDA_BUILD_ALL_TESTS)

cuda_add_executable(mspcg.x mspcg.cc main.cc wilson_dslash_reference.cpp domain_wall_dslash_reference.cpp clover_reference.cpp blas_reference.cpp)
target_link_libraries(mspcg.x ${TEST_LIBS} ${MPI_LIBRARIES} ${CUDA_LIBRARIES} cuda quda )
#target_link_libraries(invert_test ${TEST_LIBS} ${CUDA_LIBRARIES} cuda)
QUDA_CHECKBUILDTEST(mspcg.x QUDA_BUILD_ALL_TESTS)

