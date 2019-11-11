#!/bin/bash
cd ..
directory=$(pwd)
docker run -d -p 6901:6901 -p 5901:5901 -v $directory:/shared_drive intelligent3