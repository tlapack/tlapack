macro( FetchPackage pkg pkgURL gitTag )

  find_package( ${pkg} QUIET ) # Try to load ${pkg} from the system
  if( NOT ${pkg}_FOUND )

    if( EXISTS "$ENV{${pkg}_DIR}" )

      get_property( docString CACHE ${pkg}_DIR PROPERTY HELPSTRING )
      set( ${pkg}_DIR $ENV{${pkg}_DIR} CACHE STRING "${docString}" FORCE )

      add_subdirectory( ${${pkg}_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${pkg} )
      message( STATUS "Using ${pkg} from ${${pkg}_DIR}" )

    elseif( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14" )

      message( STATUS "${pkg} not found. Trying to fetch from ${pkgURL}. "
                      "It may take a while." )

      include(FetchContent)
      FetchContent_Declare(
      ${pkg}
      GIT_REPOSITORY ${pkgURL}
      GIT_TAG        ${gitTag} )

      FetchContent_MakeAvailable(${pkg})

      # Test if the fetch was successful
      if( EXISTS "${${pkg}_SOURCE_DIR}" )
      message( STATUS "Using ${pkg} from ${pkgURL}." )
      else()
      message( FATAL_ERROR "Failed in fetching ${pkg} from ${pkgURL}." )
      endif()

      # Hide ${pkg}_DIR and FETCHCONTENT_ options
      mark_as_advanced( FORCE ${pkg}_DIR )
      get_cmake_property(_variableNames VARIABLES)
      foreach (_variableName ${_variableNames})
      if( "${_variableName}" MATCHES "^FETCHCONTENT_" )
          mark_as_advanced( FORCE ${_variableName} )
      endif()
      endforeach()

    else()
      message( FATAL_ERROR "${pkg} not found. Set \"${pkg}_DIR\" to a directory containing "
                      "one of the files: ${pkg}Config.cmake, ${pkg}-config.cmake; or "
                      "containing the root of the project ${pkg}." )
    endif()

  endif( NOT ${pkg}_FOUND )
    
endmacro()