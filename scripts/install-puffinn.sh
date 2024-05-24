#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda
# If you use this file as template, don't forget to `chmod a+x newfile`

set -e

# Check for the operating system and install puffinn
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X..."
  echo "...skipping puffinn installation for unresolved compilation issues."
  #  git clone https://github.com/puffinn/puffinn.git
  #  cd puffinn
  #  python3 setup.py build
  #  pip install .
  #  cd ..
  #  rm -r puffinn

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
  # if Python3 version is one of 3.8 or 3.9 or 3.10, then install puffinn
  if [[ $(python3 --version 2>&1) == "Python 3.8"* ]] ||
     [[ $(python3 --version 2>&1) == "Python 3.9"* ]] ||
     [[ $(python3 --version 2>&1) == "Python 3.10"* ]]; then
    echo "Python3 version is below 3.11 or above. Installing puffinn."
    git clone https://github.com/puffinn/puffinn.git
    cd puffinn
    python3 setup.py build
    pip install .
    cd ..
    rm -r puffinn
  else
    echo "Python3 version is not 3.8, 3.9, or 3.10. Skipping puffinn installation."
  fi

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
