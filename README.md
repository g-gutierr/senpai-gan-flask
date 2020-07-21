# Senpai-GAN-Flask

> Aplicación Web Proyecto de fin de curso del Artificial Inteligence Developer 20 

- Anthony Cabrera - acabreragnz@gmail.com
- Gonzalo Gutiérrez - ggutierrez.ucu@gmail.com

## Contenido

- [Procedimientos Previos](#Procedimientos_Previos)
- [Instalación](#Instalación)
- [Utilización](#Utilización)
- [Referencias](#Referencias)



## Procedimientos_Previos

- Levantar una máquina virtual en Amazon AWS con sistema operativo Ubuntu 18.04 (Consultar diapositiva 2.0 del curso). Se requiere al menos 1GB de RAM y tener configurado el security group de forma que se puedan levantar conexiones HTTP y SSH.
- Levantar una sesión ssh con el user ubuntu.

## Instalación

- Hacer un clone del repositorio a la máquina de AWS `https://github.com/g-gutierr/senpai-gan-flask`

> Hacer un update de apt

```shell
sudo apt update
sudo apt upgrade
```

> Instalar Virtual Enviroment

```shell
sudo apt install python3-venv
```

> Crear y levantar un entorno virtual de python3 en la carpeta donde se bajo el repo (/home/ubuntu/senpai-gan-flask)

```shell
$ python3 -m venv venv
$ source venv/bin/activate
```

> Instalar herramientas

```shell
$ pip3 install --upgrade setuptools
$ pip3 install --upgrade pip
```

> Instalar Tensorflow y librerías necesarias

```shell
$ pip3 install tensorflow --no-cache-dir
$ pip3 install -r requirements.txt
```

> Levantar la Aplicación web de Flask

```shell
$ export FLASK_APP=app-gan
$ flask run --host=0.0.0.0
```
## Utilización

- En un navegador web ingresar a la página web dada por el servicio AWS en el puerto 5000.

**Se observará una imágen de tamaño 32x32 que es el resultado de hacer un predict en el modelo del generador previamente entrenado.
Para tener otra imagen se debe cargar nuevamente la página.**


- En caso de no haber hecho la instalación, se puede acceder al siguiente sitio y observar el funcionamiento:

http://ec2-54-210-58-217.compute-1.amazonaws.com:5000/
