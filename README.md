# Omnitext
Generic library for NLP.


# Docker 
The image used for this project is available from ``docker pull andompesta1/pytorch-dev-nlp:1.5``.


In order to run the container use the following command:
```bash
docker run -it --name onmitext --gpus all -p 6006:6006 -p 2022:22 -p 2000:8080 -p 2001:8081 -p 2002:8082 -p 2003:8083 -v /home/andompesta/Projects/data:/workspace/omnitext/data -v /home/andompesta/Projects/libraries:/workspace/libraries c387fb9631be bash
```


```bash
kill -9 $(ps aux | grep 'bash' | grep -v 'grep' | awk '{print $2}')
```