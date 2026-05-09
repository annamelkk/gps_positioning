import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def newton_solver_3d(satellites, distances, max_iter=15, tol=1e-6):

    x = np.array([
        np.mean(satellites[:, 0]),
        np.mean(satellites[:, 1]),
        np.mean(satellites[:, 2]),
        0.0  # clock bias
    ])

    for _ in range(max_iter):

        n = len(satellites)
        F = np.zeros(n)
        J = np.zeros((n, 4))

        for i, (xi, yi, zi) in enumerate(satellites):

            dx = x[0] - xi
            dy = x[1] - yi
            dz = x[2] - zi

            r = np.sqrt(dx**2 + dy**2 + dz**2)

            F[i] = r + x[3] - distances[i]

            J[i, 0] = dx / r
            J[i, 1] = dy / r
            J[i, 2] = dz / r
            J[i, 3] = 1.0

        JTJ = J.T @ J
        rhs = -J.T @ F

        delta = np.linalg.solve(JTJ, rhs)
        x = x + delta

        if np.linalg.norm(delta) < tol:
            break

    return x


# Linear Least Squares 
def linear_solver_3d(satellites, distances):

    x0, y0, z0 = satellites[0]
    d0 = distances[0]

    A = []
    b = []

    for i in range(1, len(satellites)):

        xi, yi, zi = satellites[i]
        di = distances[i]

        A.append([
            2*(xi - x0),
            2*(yi - y0),
            2*(zi - z0),
            2*(d0 - di)
        ])

        b.append(
            d0**2 - di**2 +
            xi**2 - x0**2 +
            yi**2 - y0**2 +
            zi**2 - z0**2
        )

    A = np.array(A)
    b = np.array(b)

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    return sol


# setup
satellites = np.array([
    [0, 0, 20],
    [10, 0, 15],
    [0, 10, 25],
    [10, 10, 18],
    [5, 5, 30],
    [8, 2, 22]
])

true_position = np.array([4.5, 6.2, 12.0])
true_bias = 0.3

# true pseudoranges
distances = np.linalg.norm(satellites - true_position, axis=1) + true_bias

# add noise
noise = np.random.normal(0, 0.05, size=len(distances))
noisy_distances = distances + noise


# ________________ SOLUTION __________________

newton_est = newton_solver_3d(satellites, noisy_distances)
linear_est = linear_solver_3d(satellites, noisy_distances)

print("True position:", true_position, "bias:", true_bias)
print("\nNewton estimate (x,y,z,b):", newton_est)
print("Linear estimate (x,y,z,b):", linear_est)

print("\nNewton position error:", np.linalg.norm(newton_est[:3] - true_position))
print("Linear position error:", np.linalg.norm(linear_est[:3] - true_position))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    satellites[:, 0],
    satellites[:, 1],
    satellites[:, 2],
    label="Satellites"
)

ax.scatter(
    *true_position,
    label="True Position",
    marker="x",
    s=100
)

ax.scatter(
    *newton_est[:3],
    label="Newton Estimate",
    marker="o"
)

ax.scatter(
    *linear_est[:3],
    label="Linear Estimate",
    marker="s"
)

ax.set_title("3D GPS Positioning with Clock Bias")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()


# error comparison bar plot
labels = ["Newton", "Linear"]
errors = [
    np.linalg.norm(newton_est[:3] - true_position),
    np.linalg.norm(linear_est[:3] - true_position)
]

plt.figure()
plt.bar(labels, errors)
plt.title("Positioning Error Comparison")
plt.ylabel("Error (Euclidean distance)")
plt.show()
