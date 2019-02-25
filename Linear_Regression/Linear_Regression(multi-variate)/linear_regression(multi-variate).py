import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_cost(x,y,theta):
        m=len(x)
        prod = np.subtract(np.matmul(x,theta),y)
        temp=np.square(prod)
        su = (temp.sum(axis=0))*(1/(2*m))
        return(su)
        

        
        

def gradient_descent(x,y,theta,alph,iter):
        
        xaxis=np.zeros((2000))
        yaxis=np.zeros((2000))
     
        for i in range(iter):
                prod = np.subtract(np.matmul(x,theta),y)
                m=len(prod)
                prod.shape=(1,m)
                temp = np.matmul(prod,x)
                temp.shape=(len(theta),1)
                theta=theta-(alph*1/(m))*temp
                yaxis[i]=compute_cost(x,y,theta)
                xaxis[i]=i


        return theta
        
    




def main():
        t1,t2,y=np.loadtxt('ex1data2.txt',delimiter=',',unpack=True)     
        
        me1=np.mean(t1)
        me2=np.mean(t2)
        me3=np.mean(y)

        d1=np.ptp(t1,axis=0)
        d2=np.ptp(t2,axis=0)
        d3=np.ptp(y,axis=0)
        
        m=len(t1)
        x=np.zeros((m,2))

        for i in range(m):
            x[i][0]=t1[i]=(t1[i]-me1)/d1
            x[i][1]=t2[i]=(t2[i]-me2)/d2
            y[i]=(y[i]-me3)/d3

        xdash=x
        
        x=np.insert(x,0,1,axis=1)
        y.shape=(m,1)
        
        theta = np.zeros((3,1))
        alph=0.04
        iter=2000

        
        #compute_cost(x,y,theta)                                       
        theta_val = gradient_descent(x,y,theta,alph,iter)

        print(theta_val)

        re=np.matmul(x,theta_val)
        print(re)
        print(y)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(t1 , t2 ,y, alpha=1.0, c='green',marker='.')
        ax.scatter3D(t1, t2, re,alpha=1.0, c='red',marker='x')
        plt.show()
        
 
if __name__ == "__main__":
    main()
