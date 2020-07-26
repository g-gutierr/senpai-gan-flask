# Proyecto AI Dev Senpai 2020

Proyecto de fin de curso del Artificial Inteligence Developer 20

## Contenido

- [Conformación del equipo](##Conformación%20del%20equipo)
- [Descripción de la problemática a solucionar](##Descripción-de-la-problemática-a-solucionar)
- [Descripción de la solución implementada](## Descripción de la solución implementada)
- [Descripción inicial del algoritmo de machine learning o modelo de deep learning a utilizar](## Descripción_inicial_del_algoritmo_de_machine_learning_o_modelo_de_deep_learning_a_utilizar)
- [Análisis de soluciones existentes y detalle de la alternativa seleccionada](## Análisis%20de%20soluciones%20existentes%20y%20detalle%20de%20la%20alternativa%20seleccionada)
- [Resultados obtenidos](## Resultados%20obtenidos)
- [Propuestas a futuro](## Propuestas%20a%20futuro)
- [Correr el proyecto](## Correr%20el%20proyecto)
- [Referencias y Bibliografía](#referencias-y-bibliografía)


## Conformación del equipo

- Anthony Cabrera - acabreragnz@gmail.com
- Gonzalo Gutiérrez - ggutierrez.ucu@gmail.com

## Descripción de la problemática a solucionar

En el marco del proyecto se entrenará una red *Generative Adversarial Network* (GAN) [1] con el fin de generar imágenes a color de frentes de automóviles con una resolución de 32 por 32 píxeles. Con el fin de disponibilizar el generador, se proveerá un endpoint por medio de la implementación de una API utilizando el *framework web* Flask [2].

## Descripción de la solución implementada

Las redes GAN son una clase de algoritmo de *deep learning* el cual consiste de dos redes neuronales, el generador y el discriminador las cuales compiten entre sí buscando mejorar su desempeño de generación y detección respectivamente.
Por un lado el generador a partir de una entrada de ruido aleatorio genera una muestra falsa. Mientras que el discriminador por otra parte intenta distinguir las muestras reales (del conjunto de entrenamiento), de las falsas que son generadas por la red generadora. Esta competencia lleva a que el discriminador aprenda a clasificar correctamente los ejemplos como reales o falsos y simultáneamente el generador sea capaz de generar muestras más cercanas a la realidad y lograr así engañar a la red discriminadora.

Para este proyecto se diseña y entrena una red basada en la arquitectura *Deep Convolutional Generative Adversarial Network*(DCGAN) [3] ya que las mismas tienen un mejor desempeño para las imágenes por el aprovechamiento de la información espacial. En la misma el generador se encargará de generar imágenes de automóviles de un tamaño de 32x32x3 en un comienzo, y el discriminador de clasificar si las muestras tanto del dataset como las producidas por el generador son reales o falsas.

En cuanto al Dataset para entrenar a la red neuronal llamado *"The Comprehensive Cars (CompCars) dataset"* [4] que tiene un total de 136.726 imágenes de autos tomadas desde diferentes ángulos. El origen de las mismas fue realizado a partir de un scraping de la web (*web-nature*), así como también, tomadas de cámaras en la vía pública (*surveillance-nature*). En nuestro caso, se utilizarán las 44.481 imágenes de frentes de autos que forman parte del sub-set de surveillance-nature.
Es importante resaltar que se hace un procesado de las imágenes ya que las mismas tienen diversas resoluciones. Por efectos prácticos se llevan las mismas a un tamaño de 32x32x3 con el fin de reducir los tiempos de entrenamiento. Por otra parte, se realiza también el escalado de los valores de los pixeles al rango [-1, 1], ya que es recomendado usar la activación hyperbolic tangent (tanh) como salida del modelo generador [5].

## Descripción inicial del algoritmo de machine learning o modelo de deep learning a utilizar

Cómo se mencionó anteriormente la solución planteada consiste en una red DCGAN la cual pasaremos a detallar a continuación:

### Red Generador:
---

Se comienza con un vector de ruido (z) que será el punto de entrada a las capas iniciales que generan imágenes de baja resolución. Progresivamente se aumenta la resolución con capas posteriores hasta llegar a tener a la salida una resolución cuyo *output* G(z) tiene la misma dimensión que los elementos post-procesados del *dataset* (32x32x3).

* **Entrada**: Vector de ruido (z) de largo 100.
* **Salida**:  Imágenes con tres canales de colores y de tamaño 32 por 32 píxeles.

A partir de lo visto en diversa literatura [5],[6],[7], típicamente la arquitectura para un generador en una DCGAN se compone de lo siguiente:

* Seteo de hiper parámetros para el vector de ruido 

```python
noise_dim = 100
```
* Se define la base para la generación de una imágen de 4x4 y 256 nodos como para poder generar múltiples versiones de la imágen de salida.

```python
generator = Sequential()
n_nodes = 4 * 4 * 256 
generator.add(Dense(n_nodes, input_dim=noise_dim))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((4, 4, 256)))
```

* *Upsample* a 8x8. Se aplica la capa *Conv2DTranspose*, con un *stride*=(2,2) cuadruplicando el tamaño de la imágen, y un *kernel_size*=(4,4) múltiplo del *stride*.

```python
generator.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
```
* *Upsample* a 16x16

```python
generator.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
```
* *Upsample* a 32x32

```python
generator.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
```
* El *upsample* se ha realizado, ahora se procede a agregar una capa *Conv2DTranspose*, para los 3 canales a color, dando a la salida una imagen de 32x32, normalizados para la entrada al modelo del discriminador a partir de la función de activación *tanh*.

```python
generator.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
```

Finalmente, el código para el modelo del generador, queda de la siguiente manera:

```python
noise_dim = 100
generator = Sequential()
n_nodes = 4 * 4 * 256 
generator.add(Dense(n_nodes, input_dim=noise_dim))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((4, 4, 256)))
generator.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv2DTranspose(128, (4,4), strides = (2,2), padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))
generator.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
```

![Arquitectura Generador](public/arquitectura-generador.png?raw=true "Fig. 1: Ejemplo de arquitectura del Generador para una DCGAN.")

**Fig. 1:** Ejemplo de arquitectura del Generador para una DCGAN.

A partir de fuentes consultadas, se observan recomendaciones a tener en cuenta para la red generador:

* Eliminar capas *Fully Connected* [3].
* Utilizar *batch normalization* en todas las capas excepto la capa de salida [3].
* En las capas de *Conv2DTranpose* para el upsampling, utilizar *kernels* con tamaño múltiplo del tamaño del stride en el generador, esto para solucionar el problema de *checkerboard artifacts* que sucede al generar imágenes [8].


### Red Discriminador:
--- 

La red discriminador consiste en un clasificador CNN, con ciertas modificaciones mencionadas a continuación para un mejor desempeño del mismo dentro de la GAN.

* **Entrada**: Imágenes con tres canales de colores y de tamaño 32 por 32 píxeles.
* **Salida**: Clasificación binaria, se utiliza 1 nodo cuya activación es la probabilidad de que la imagen sea real o falsa.

![Arquitectura Discriminador](public/arquitectura-discriminator.png?raw=true "Fig. 2: Ejemplo de arquitectura del Discriminador para una DCGAN.")

**Fig. 2:** Ejemplo de arquitectura del Discriminador para una DCGAN.

A continuación se presenta una versión inicial del modelo discriminador:

```python
discriminator = Sequential()
discriminator.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
```
* Se hace un *downsample* a 16 x 16, y se duplican la cantidad de filtros

```python
discriminator.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
```
* Se hace un *downsample* a 8 x 8

```python
discriminator.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
```

* Se hace un *downsample* a 16 x 16, y se duplican la cantidad de filtros

```python
discriminator.add(Conv2D(256, (4,4), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))
```

* Se define un clasificador 

```python
discriminator.add(Flatten())
discriminator.add(Dropout(0.4))
discriminator.add(Dense(1, activation='sigmoid'))
```

* Finalmente se compila

```python
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

Se usa como optimizador *Adam*, con *learning rate* = 0.2x10-3 y *momentum* = 0.5.
Como función de pérdida se utiliza *binary_crossentropy*, tras tratarse de un problema de clasificación binaria.

Juntando las partes anteriores, el discriminador queda definido de la siguiente manera:

```python
discriminator = Sequential()
discriminator.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))

discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

discriminator.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

discriminator.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

discriminator.add(Conv2D(256, (4,4), strides=(2, 2), padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

discriminator.add(Flatten())
discriminator.add(Dropout(0.4))
discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

Se observaron las siguientes recomendaciones a tener en cuenta para la red discriminador:

* Reemplazar todas las capas de *pooling* por capas convolucionales con *stride* [5].
* Usar *batch normalization* en todas las capas ocultas, antes de la activación *LeakyReLU* [5]. Esto acarrea ciertos problemas [9] que se solucionan entrenando el discriminador separadamente en imágenes reales por un lado, y falsas por otro.
* Usar la activación *LeakyReLU* para todas las capas (a excepción de la salida) con el valor recomendado 0,2 [5].
* En las capas de *Conv2D* utilizar *kernels* con tamaño múltiplo del tamaño del *stride* en el discriminador, esto para solucionar el problema de *checkerboard artifacts* que sucede al generar imágenes [8].
* Utilizar una capa de *Dropout* en el clasificador [8].


### Arquitectura global de la solución:
---
Se tiene una API implementada en flask, la cual provee un endpoint [localhost:5000/example1.png](http://localhost:5000/example1.png) encargado de generar una nueva imagen de un auto de frente con resolución 32x32. Al levantar el servidor, se carga el modelo `model/model_gen.h5` de la red generador definida y entrenada en la google colab `Proyecto_Senpai_AI_dev_2020.ipynb` la cual se encuentra en la raíz de este proyecto. Y luego para cada llamada al endpoint `example1.png` se ejecuta el método `predict` del modelo, el cual genera una nueva imagen y lo envía como respuesta en el pedido HTTP.

## Análisis de soluciones existentes y detalle de la alternativa seleccionada

Se encontró una publicación de Jason Brownlee, titulada *How to Develop a DCGAN for Small Color Photographs*, en la misma desarrollan la solución al problema de generar imágenes a color de pequeña resolución utilizando una DCGAN, la red es entrenada utilizando el dataset CIFAR-10. En el marco de este proyecto se realizan cambios en la arquitectura como el uso de batch normalization, el entrenamiento del discriminador sin mezclar imágenes falsas con verdaderas, y además para la entrega final se considera aplicar suavizado de las etiquetas, como también el uso de kernels de tamaño múltiplo del tamaño de los strides, etc. <br />
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

También se encontró un repositorio de GitHub, donde se aborda una problemática similar. En este caso hacen un *scraping* a partir de dos dos sitios web de venta de autos (Carvana & Carmax). El link a su repositorio es el siguiente: <br /> https://github.com/asoomar/car-design-generation

En cuanto a datasets, existe un dataset conteniendo 16.000 imágenes de autos, llamado *"Cars Dataset"* el cual se encuentra disponible en la siguiente dirección: <br />
https://ai.stanford.edu/~jkrause/cars/car_dataset.html

## Resultados obtenidos
Los resultados obtenidos son buenos, a simple vista y más allá de pequeños detalles, los autos generados son parecidos a la realidad:

![Resultados](public/resultados.png?raw=true)

En cuanto a la perdida de ambos modelos, parecen resultados razonables los cuales se mantienen estables con tendencia a decrecer a medida que aumentan las epocas:

![Resultados-perdida](public/resultados-perdida.png?raw=true)

Se puede ver  el progreso de los resultados en el entrenamiento en detalle en la sección *6. Entrenar el GAN* de la google colab `Proyecto_Senpai_AI_dev_2020.ipynb`.

## Propuestas a futuro
Quedan fuera del alcance de este proyecto algunas mejoras planificadas en un principio las cuales serían de interes para mejorar la generación de imagenes reales por parte de la red generadora. Algunos de ellos son:

- Preentrenar el discriminador
- Hacer un suavizado de etiquetas (para algunos ejemplos usar valores cercanos a 0 y 1)
- Agregar ruido al dataset (para algunas imagenes reales etiquetarlas con la clase falsa, para algunas imagenes falsas etiquetarlas con la clase verdadera.)
- Utilizar *latent space* de distinto tamaño
- Agregar capas ocultas en ambas redes para hacer *up sampling* y *down sampling* de manera más progresiva.
- Tunear hiperparámetros de ambas redes (cambiar la cantidad de nodos de las capas, cambiar el optimizador, *batch size*, cantidad de *epochs*, etc).

## Correr el proyecto

### Procedimientos previos

- Levantar una máquina virtual en Amazon AWS con sistema operativo Ubuntu 18.04 (Consultar diapositiva 2.0 del curso). Se requiere al menos 1GB de RAM y tener configurado el security group de forma que se puedan levantar conexiones HTTP y SSH.
- Levantar una sesión ssh con el user ubuntu.

### Instalación

- Hacer un clone del repositorio a la máquina de AWS.

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
### Utilización

- En un navegador web ingresar a la página web dada por el servicio AWS en el puerto 5000.

**Se observará una imágen de tamaño 32x32 que es el resultado de hacer un predict en el modelo del generador previamente entrenado.
Para tener otra imagen se debe cargar nuevamente la página.**

### Referencias de Implementación
---
1 - Creación entornos virtuales y Flask - https://linuxize.com/post/how-to-install-flask-on-ubuntu-18-04/<br />
2 - Documentación Routing - https://flask.palletsprojects.com/en/1.1.x/quickstart/#routing<br />
3 - Cache Response - https://github.com/davewsmith/play/tree/master/matplotlib-flask

## Referencias y Bibliografía
[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.<br />
[2] https://palletsprojects.com/p/flask/<br />
[3] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).<br />
[4]  Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, In Computer Vision and Pattern Recognition (CVPR), 2015.<br />
[5] Brownlee, Jason. Generative Adversarial Networks with Python: Deep Learning Generative Models for Image Synthesis and Image Translation. Machine Learning Mastery, 2019.<br />
[6] https://medium.com/analytics-vidhya/implementing-a-gan-in-keras-d6c36bc6ab5f <br />
[7] https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0 <br />
[8] François, Chollet. "Deep learning with Python." (2017).<br />
[9] https://datascience.stackexchange.com/questions/56860/dc-gan-with-batch-normalization-not-working<br />

