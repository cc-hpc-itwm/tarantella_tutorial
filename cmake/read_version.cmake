include (CMakeParseArguments)

function (configure_version)
  set(one_value_options VERSION_FILE)
  set(required_options VERSION_FILE)
  cmake_parse_arguments (ARG "${options}" "${one_value_options}"
                        "${multi_value_options}" "${required_options}" ${ARGN})

  file(READ ${ARG_VERSION_FILE} ver)
  string(REGEX MATCH "tnt_version[ ]*=[ ]*[\'\"]([0-9]+\.[0-9]+\.?[0-9]*)[\'\"]" _ ${ver})
  if(CMAKE_MATCH_COUNT EQUAL 1)
    set(TNT_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
    message(STATUS "Tarantella-tutorial version: ${CMAKE_MATCH_1}")
  else()
    message(FATAL_ERROR "Invalid version string in ${ARG_VERSION_FILE}" )
  endif()
endfunction()