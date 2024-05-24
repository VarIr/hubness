#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda
# If you use this file as template, don't forget to `chmod a+x newfile`

set -e

# Check for the operating system and install nmslib
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X..."
  git clone https://github.com/nmslib/nmslib.git
  cd nmslib/python_bindings
  python3 -m pip install .
  cd ../..
  rm -r nmslib

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux..."
  pushd /tmp
  git clone https://github.com/nmslib/nmslib.git
  pushd nmslib/python_bindings
  python3 -m pip install .
  popd
  rm -r nmslib
  popd

elif [[ $(uname -s) == MINGW32_NT* ]]; then
  echo "Running under Win x86-32"
  echo "Nothing to build."

elif [[ $(uname -s) == MINGW64_NT* ]]; then
  echo "Running under Win x86-64"
  echo "Nothing to build."

elif [[ $(uname -s) == CYGWIN* ]]; then
  echo "Running under Cygwin"
  echo "Nothing to build."

fi
