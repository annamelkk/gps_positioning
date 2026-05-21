###  GPS Positioning via Nonlinear Equations

  


1. Background

GPS works by using a constellation of over 30 satellites orbiting Earth, which transmit radio signals containing their exact timer and position [1](https://spaceplace.nasa.gov/gps/en/). GPS devices use trilateration, which require at least 4 satellites to determine the position of a point.

1 satellite places the point somewhere on a sphere, 2 satellites narrows the position to a circle, 3 satellites narrows the position to two possible points and 4 satellites validate information, resolving time differences between the satellite and receiver clocks [2](https://prezi.com/p/e9rjbtx03ftq/trilateration-in-gps/).

**Problem Statement:** Implement Newton’s method for estimating a position using data from 4-6 GPS satellites.

2. Mathematical Solution

Using trilateration each satellite creates an equation of form 

$$(x-x_i)^2+(y - y_i)^2 = r^2_i \ \ \ \ \ \text{(2D case)}$$

In this case 3 satellites will give a unique solution.

$$(x-x_i)2+(y - y_i)^2+(z-z_i)^2 - c\cdot \text{d}T   = r_i \ \ \ \ \ \text{(3D case)}$$

So, each satellite observes circular (spherical for 3D) area (volume) and the intersections of these circles (spheres) help to find true coordinates.

To solve a system of equations consisting of data from each satellite Newton’s and Least Squares methods will be implemented.

  

**Newton’s Method**

A function $F(x, y) = [f1(x, y), ... , fn(x,y)]$, where 

$$f_i(x,y)=(x-x_i)^2+(y - y_i)^2 - r^2_i$$

Given a starting guess $v_0 = (x_0, y_0)$ for solving $F(x, y) = 0$, the solution can be found using 

$$v_{(n+1)} = v_n - J^{-1}(v_n)F(v_n)$$

Where J is the Jacobian matrix (and $J^{-1}$ is the inverse Jacobian). 

However, inverse calculation requires computational resources.

For an n n matrix computing inverse requires about $2n^3$ flops. In comparison solving a linear system of equations requires only about  $\frac{2}{3}n^3+2n^2$ with no additional multiplications [3](https://gregorygundersen.com/blog/2020/12/09/matrix-inversion/#:~:text=Inverting%20a%20matrix&text=But%20the%20basic%20idea%20is,which%20effectively%20solves%20for%20x.&text=As%20we%20can%20see%20inverting,as%20directly%20solving%20for%20x.).


So instead, for our overdetermined case (more satellites than unknowns) we will use the following algorithm. Take the equation 


$$J(x^{(k-1)})y^k=-F(x^{(k-1)})$$
  

Since there are 2 unknowns but 4-6 satellites, J is not a square matrix, so we have to normalize it


$$(JTJ)y=-JTF$$

  

In Python the solution will look like this: 

  

```python
JTJ = J.T @ J  
rhs = -J.T @ F  
y = np.linalg.solve(JTJ, rhs)  
  
x = x + y
```
  

**Least Squares Method**  

When more than three satellites are available, the system becomes overdetermined and cannot be solved exactly due to measurement noise. We therefore apply the least squares method, which finds the position that minimizes the squared residual error 


$$\text{min} ||Ax - b||^2$$

This leads to normal equations 


$$A^TAx=A^T b,$$

whose solution provides the best estimate of the receiver position  

This is implemented using Python 

```python
sol = np.linalg.lstsq(A, b, rcond=None)[0]
```

