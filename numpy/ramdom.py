import numpy as np

x = np.random.rand(5)   #picks the random number between from 0 to 1
print(x)

x = np.random.randn(5) #it choses the number very close to 0 and can be negative as well
print(x)

#we can define our own range and then can pick some random numbers between them
x = np.random.randint(0,100,10)
print(x)