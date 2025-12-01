if make any changes in code or files :
     docker build -t deepshot . 
docker run -it --rm -p 8501:8501 -v /Users/codemonkey/Downloads:/data deepshot:latest