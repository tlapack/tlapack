# Copyright (c) 2025, University of Colorado Denver. All rights reserved.
#
# This file is part of <T>LAPACK.
# <T>LAPACK is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)

  set(DOXYGEN_PROJECT_NAME           "<T>LAPACK")
  set(DOXYGEN_JAVADOC_AUTOBRIEF      YES)
  set(DOXYGEN_QT_AUTOBRIEF           YES)
  set(DOXYGEN_TOC_INCLUDE_HEADINGS   0)
  set(DOXYGEN_EXTRACT_STATIC         YES)
  set(DOXYGEN_CASE_SENSE_NAMES       YES)
  set(DOXYGEN_SORT_BRIEF_DOCS        YES)
  set(DOXYGEN_LAYOUT_FILE            ${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml)
  set(DOXYGEN_WARN_IF_UNDOCUMENTED   NO)
  set(DOXYGEN_WARN_LOGFILE           ${CMAKE_CURRENT_BINARY_DIR}/doxygen_warnings.log)
  set(DOXYGEN_FILE_PATTERNS          "*.c" "*.cc" "*.cxx" "*.cpp" "*.c++" "*.h"
                                     "*.hh" "*.hxx" "*.hpp" "*.h++" "*.f90"
                                     "*.f95" "*.f03" "*.f08" "*.f")
  set(DOXYGEN_RECURSIVE              YES)
  set(DOXYGEN_INPUT_FILTER           ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-filter.pl)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${PROJECT_SOURCE_DIR}/README.md)
  set(DOXYGEN_GENERATE_TREEVIEW      YES)
  set(DOXYGEN_USE_MATHJAX            YES)
  set(DOXYGEN_MATHJAX_RELPATH        https://cdn.jsdelivr.net/npm/mathjax@2)
  set(DOXYGEN_MATHJAX_EXTENSIONS     TeX/AMSmath)
  set(DOXYGEN_LATEX_CMD_NAME         latex)
  set(DOXYGEN_EXTRA_PACKAGES         amsmath)
  set(DOXYGEN_MACRO_EXPANSION        YES)
  set(DOXYGEN_EXPAND_ONLY_PREDEF     YES)
  set(DOXYGEN_PREDEFINED             "TLAPACK_CHECK_INPUT"
                                     "__cplusplus=202002L")
  set(DOXYGEN_EXCLUDE_SYMBOLS        "isSlice")
  set(DOXYGEN_HAVE_DOT               YES)

  doxygen_add_docs(
    doxygen
    ${CMAKE_CURRENT_SOURCE_DIR}/groups.dox
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/test/include
    COMMENT "Generate documentation"
  )

else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)