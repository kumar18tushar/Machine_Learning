import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_costFunction(x,y,theta):
    m=len(x)
    prod = np.subtract(np.matmul(x,theta),y)
    temp=np.square(prod)
    sum_matrix = (temp.sum(axis=0))*(1/(2*m))
    return sum_matrix
    



def gradient_descent(x,y,theta,alpha,iter):
    for i in range(iter):
        m=len(x)
        prod1 = np.subtract(np.matmul(x,theta),y)
        l=len(prod1)
        prod1.shape=(1,l)
        prod2=np.matmul(prod1,x)
        prod2.shape=(2,1)
        theta=theta-(alpha*1/(m))*prod2
        cost=compute_costFunction(x,y,theta)
    return theta
    


def visualise(x,y):
    thetaval0 = np.linspace(-10,10,num=100)
    thetaval1 = np.linspace(-1,4,100)
    jval=np.zeros((100,100),dtype=int)

    for i in range(len(thetaval0)):
        for j in range(len(thetaval1)):
            th=np.array([thetaval0[i],thetaval1[j]])
            th.shape=(2,1)
            b=compute_costFunction(x,y,th)
            jval[i][j]=b

    np.transpose(jval)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.plot_surface(ax,thetaval0, thetaval1, jval)
    plt.show()




def show_plot(x,y):
	plt.scatter(x , y , alpha=0.8, c='red',marker='x')
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in $10,000s')
	#plt.show() 

	

def main():
        x, y = np.loadtxt('ex1data1.txt',delimiter=',',unpack=True)         
        m=len(x)
        x.shape =(m,1)                                                      
        xtemp=x
        y.shape=(m,1)
        x=np.insert(x,0,1,axis=1)                                           

        theta_init = np.array([0,0])
        theta_init.shape=(2,1)
        alpha=0.01
        iter=1500
        
        theta_val = gradient_descent(x,y,theta_init,alpha,iter)

        re=np.matmul(x,theta_val)
        visualise(x,y)
   

     
if __name__ == "__main__":
    main()
