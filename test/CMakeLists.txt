#-------------------------------------------------------------------------------
# Search for Catch2
find_package( Catch2 QUIET )
if( NOT Catch2_FOUND )
  find_package( Catch2 REQUIRED
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty" )
  message( STATUS "  Using Catch2 from ${Catch2_DIR}" )
endif()

add_subdirectory(blas)