import numpy as np
from skimage.io import imread
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter


#el tamaño máximo ocurre a las 30 segundos, despues aumenta el número de semillas. 

#cargar imagen
imagen = imread('Semilla30.png', True)
imagen = np.interp(imagen, [np.min(imagen), np.max(imagen)], [0, 255])


plt.ion()
fig1 = plt.figure('Gráfica')
fig2 = plt.figure('Evolución Level Set')

#generar figura
def img(v: np.ndarray, imagen: np.ndarray):
    fig2.clf()
    contours = measure.find_contours(v, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(imagen, interpolation='nearest', cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], color='r', linewidth=3)

#generar gráfica
def grfc(v: np.ndarray):
    fig1.clf()
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = v.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -v, rstride=2, cstride=2, color='b', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, v, 0, colors='g', linewidths=2)



#función que dibuja la ecuación level set
#pause: controla la velocidad de la animación

def drawLS(v: np.ndarray, imagen: np.ndarray, pause=0.003):
    
    img(v, imagen)
    grfc(v)
    plt.pause(pause)


#Metodol Level Set

#v0: level set inicial que se actualizará
#g: indicador de borde (23)
#lamda: coeficiente del término de longitud
#mu: coeficiente de la distancia de regularización
#alfa: coeficiente del término de área ponderado
#epsilon: coeficiente  delta de Dirac
#timestep: paso de tiempo
#iteras: número de iteraciones   


#Indicador de borde (ecuación 23)

def Borde(v0, g, lamda, mu, alfa, epsilon, timestep, iteras=100000): 
   
    v = v0.copy()
    [vy, vx] = np.gradient(g)
    for k in range(iteras):
        v = neumann(v)
        [v_y, v_x] = np.gradient(v)
        s = np.sqrt(v_x**2 + v_y**2)
        delta = 1e-10 #modificación de la ec original donde delta = 1 (causa problemas por la poca "suavidad" de la imagen)
        n_x = v_x / (s + delta)  # Delta agrega un numero muy pequeño positivo, para evitar cero en el denominador
        n_y = v_y / (s + delta)
        curvatura = div(n_x, n_y)
        
        
        dirac_v = dirac(v, epsilon)
        area_term = dirac_v * g 
        edge_term = dirac_v * (vx * n_x + vy * n_y) + dirac_v * g * curvatura
        v += timestep * (mu *DoubleWell(v)  + lamda * edge_term + alfa * area_term)

        #Agregar un arreglo vacio (más arriba) y aqui un contador que guarde los v. para graficar despues
    return v


#Función potencial, Calcular el termino de regulación de distancia con el potencial double well (ecuación 16)
def DoubleWell(v):
    
    [v_y, v_x] = np.gradient(v)
    s = np.sqrt(np.square(v_x) + np.square(v_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)

    #calcular la primera derivada de la función potencial (ecuación 16)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    #calcular ecuación (10). Cuando s tiende a cero, d_p(s) tiende a 1, (ecuación 18)   
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0)) 
    return div(dps * v_x - v_x, dps * v_y - v_y) + laplace(v, mode='nearest')


#divergencia
def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy


#delta Dirac
def dirac(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b

#Condición de frontera de Neumann
def neumann(f):    
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g


def LevelSet(imagen: np.ndarray, iniLSF: np.ndarray, timestep, iter, iterMax, lamda,
             alfa, epsilon, sigma =1):
    
   

    # coeficiente del termino de distancia de regularización R(v)
    mu = 0.2/ timestep  

    imagen = np.array(imagen, dtype='float32')

    #suavizar imagen con filtro gaussiano
    imagensuav = gaussian_filter(imagen, sigma)
    [Iy, Ix] = np.gradient(imagensuav)
    f = np.square(Ix) + np.square(Iy)
    
    #indicador de borde (ecuación 23)
    g = 1 / (1 + f) 

    #inicializar LSF 
    v = iniLSF.copy()

    img(v, imagen)
    grfc(v)
    
    print('Evolución Level Set en el tiempo')

    
    #Iniciar la evolucion de la función Level Set
    for n in range(iterMax):
        v = Borde(v, g, lamda, mu, alfa, epsilon, timestep, iter)
        print('Level set en el segundo %i' % n)
        drawLS(v, imagen)

    
    #refinar el contorno de nivel cero  mediante alfa= 0
    alfa = 0
    iteraref = 10
    v = Borde(v, g, lamda, mu, alfa, epsilon, timestep, iteraref)
    return v



def Parametros():    

    # inicializar la LSF como una función de paso binaria (ecuación 19)
    c0 = 2
    iniLSF = c0 * np.ones(imagen.shape)
    
    #generar la región inicial R0
    iniLSF[28:33, 48:53] = -c0
    
    # parametros
    return {
        'imagen': imagen,
        'iniLSF': iniLSF,
        'timestep': 5,
        'iter': 10,
        'iterMax': 30,
        'lamda': 5,  #coeficiente del largo L(v)
        'alfa': -3,  #coeficiente del termino de area A(v)
        'epsilon': 1.5,  #parámetro para el delta de Dirac
            }



params = Parametros()
v = LevelSet(**params)

print('Final')
drawLS(v, params['imagen'], 10)


