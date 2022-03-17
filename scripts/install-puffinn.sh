#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda
# If you use this file as template, don't forget to `chmod a+x newfile`

set -e

# Check for the operating system and install puffinn
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X..."
  git clone https://github.com/puffinn/puffinn.git
  cd puffinn
  python3 setup.py build
  pip install .
  cd ..
  rm -r puffinn

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux..."
  # Trying to install puffinn from cache,
  # and only build if this fails.
  #  pip install puffinn || (\
  #    git clone https://github.com/puffinn/puffinn.git;\
  #    cd puffinn;\
  #    python3 setup.py build;\
  #    pip install . ;\
  #    cd ..)
  git clone https://github.com/puffinn/puffinn.git
  cd puffinn
  python3 setup.py build
  pip install .
  cd ..
  rm -r puffinn

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
