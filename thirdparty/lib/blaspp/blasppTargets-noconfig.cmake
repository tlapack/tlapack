#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "blaspp" for configuration ""
set_property(TARGET blaspp APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(blaspp PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libblaspp.so"
  IMPORTED_SONAME_NOCONFIG "libblaspp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS blaspp )
list(APPEND _IMPORT_CHECK_FILES_FOR_blaspp "${_IMPORT_PREFIX}/lib/libblaspp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
