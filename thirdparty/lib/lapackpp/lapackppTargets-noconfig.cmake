#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lapackpp" for configuration ""
set_property(TARGET lapackpp APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lapackpp PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liblapackpp.so"
  IMPORTED_SONAME_NOCONFIG "liblapackpp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS lapackpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_lapackpp "${_IMPORT_PREFIX}/lib/liblapackpp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
