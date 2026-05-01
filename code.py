import numpy as np
import matplotlib.pyplot as plt

satellites = np.array([
    [0,  0],
    [7,  0],
    [0,  7],
    [7,  7],
    [5, 15],
    [15,15]
])

true_position = np.array([5.6, 7.8])

# distances
distances = np.linalg.norm(satellites - true_position, axis=1)

# random noise
noise = np.random.normal(0, 0.05, size=len(distances))
noisy_distances = distances + noise


def newton_solver(sat, d, max_iter=20):
    x = np.mean(sat, axis=0)  # initial guess

    for _ in range(max_iter):

        F = np.zeros(len(sat))
        J = np.zeros((len(sat), 2))

        for i, (xi, yi) in enumerate(sat):
            F[i] = (x[0]-xi)**2 + (x[1]-yi)**2 - d[i]**2
            J[i, 0] = 2*(x[0] - xi)
            J[i, 1] = 2*(x[1] - yi)

        delta = np.linalg.lstsq(J, F, rcond=None)[0]
        x = x - delta

    return x


def linear_solver(sat, d):
    x1, y1 = sat[0]
    d1 = d[0]

    A = []
    b = []

    for i in range(1, len(sat)):
        xi, yi = sat[i]

        A.append([
            2*(xi - x1),
            2*(yi - y1)
        ])

        b.append(
            d1**2 - d[i]**2 +
            xi**2 - x1**2 +
            yi**2 - y1**2
        )

    A = np.array(A)
    b = np.array(b)

    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol


# solve
newton_est = newton_solver(satellites, noisy_distances)
linear_est = linear_solver(satellites, noisy_distances)

# errors
error_newton = np.linalg.norm(newton_est - true_position)
error_linear = np.linalg.norm(linear_est - true_position)

print("True position:", true_position)
print("Newton estimate:", newton_est)
print("Linear estimate:", linear_est)

print("\nNewton error:", error_newton)
print("Linear error:", error_linear)


plt.scatter(satellites[:,0], satellites[:,1], label="Satellites")
plt.scatter(*true_position, label="True", marker="x", s=100)
plt.scatter(*newton_est, label="Newton", marker="o")
plt.scatter(*linear_est, label="Linear", marker="s")

plt.legend()
plt.title("GPS Positioning")
plt.axis("equal")
plt.show()



# exp2

sat_counts = [3, 4, 5, 6]

errors_newton = []
errors_linear = []

for k in sat_counts:

    sat_subset = satellites[:k]

    # recompute distances for subset
    d = np.linalg.norm(sat_subset - true_position, axis=1)

    # add noise
    noise = np.random.normal(0, 0.05, size=len(d))
    noisy_d = d + noise

    # solve
    est_newton = newton_solver(sat_subset, noisy_d)
    est_linear = linear_solver(sat_subset, noisy_d)

    # compute errors
    err_newton = np.linalg.norm(est_newton - true_position)
    err_linear = np.linalg.norm(est_linear - true_position)

    errors_newton.append(err_newton)
    errors_linear.append(err_linear)


# plot
plt.figure()
plt.plot(sat_counts, errors_newton, marker='o', label="Newton")
plt.plot(sat_counts, errors_linear, marker='s', label="Least Squares")

plt.xlabel("Number of satellites")
plt.ylabel("Position error")
plt.title("GPS Error vs Number of Satellites")

plt.legend()
plt.grid()
plt.show()
