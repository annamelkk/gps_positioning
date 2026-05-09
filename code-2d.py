import numpy as np
import matplotlib.pyplot as plt

# _____________ SETUP ___________

# manually selecting coordinates of the satellites
satellites = np.array([
    [0,   0],
    [10,  0],
    [0,  10],
    [10, 10],
    [5,   2],
    [8,   9]
])

# true receiver position
true_position = np.array([4.5, 6.2])

distances = []

for xi, yi in satellites:

    distance = np.sqrt(
        (true_position[0] - xi)**2 +
        (true_position[1] - yi)**2
    )

    distances.append(distance)

distances = np.array(distances)


# Gaussian noise to simulate real GPS 
noise = np.random.normal(0, 0.05, size=len(distances))
noisy_distances = distances + noise


print("True distances:", distances)
print("Noisy distances:", noisy_distances)


# ____________ NEWTON SOLVER ____________
def newton_solver(satellite, distances, max_iter=20, tol=1e-6):

    x = np.mean(satellite, axis=0)

    for _ in range(max_iter):

        F = np.zeros(len(satellite))
        J = np.zeros((len(satellite), 2))

        for i, (xi, yi) in enumerate(satellite):

            dx = x[0] - xi
            dy = x[1] - yi

            # for each satellite, computing how far off 
            # the current estimate is
            F[i] = dx**2 + dy**2 - distances[i]**2
            
            # J consists of partial deriatives of F
            J[i, 0] = 2 * dx
            J[i, 1] = 2 * dy
        
        # solving the equation JT J * delta = -JT F
        # JT J projects into the column space of J effectively
        # making the overdetermined matrix usable for solving
        JTJ = J.T @ J
        rhs = -J.T @ F

        delta = np.linalg.solve(JTJ, rhs)

        x = x + delta
        #repeat until ||delta|| < 10^-6 or max iterations reached
        if np.linalg.norm(delta) < tol:
            break

    return x


# ______________ LINEAR LEAST SQUARES METHOD _________

def linear_solver(satellites, distances):

    x1, y1 = satellites[0]
    d1 = distances[0]

    A = []
    b = []

    for i in range(1, len(satellites)):

        xi, yi = satellites[i]
        di = distances[i]

        # we start from (x-xi)2 + (y-yi)2 = di2
        # for obtaining linear equation for this method
        # we just subtract eq for sat 0 from every other eq

        A.append([2*(xi - x1), 2*(yi - y1)])

        b.append(
            d1**2 - di**2 +
            xi**2 - x1**2 +
            yi**2 - y1**2
        )

    A = np.array(A)
    b = np.array(b)

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    return sol

# Solve
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


# Plots
plt.scatter(satellites[:,0], satellites[:,1], label="Satellites")
plt.scatter(*true_position, label="True", marker="x", s=100)
plt.scatter(*newton_est, label="Newton", marker="o")
plt.scatter(*linear_est, label="Linear", marker="s")

plt.legend()
plt.title("GPS Positioning")
plt.axis("equal")
plt.grid()
plt.show()



# _____________ SATELLITE COUNT INFLUENCE ___________

sat_counts = [3, 4, 5, 6]

errors_newton = []
errors_linear = []

for k in sat_counts:

    sat_subset = satellites[:k]

    # recompute distances
    d = np.linalg.norm(sat_subset - true_position, axis=1)

    # noise
    noise = np.random.normal(0, 0.05, size=len(d))
    noisy_d = d + noise

    # solve
    est_newton = newton_solver(sat_subset, noisy_d)
    est_linear = linear_solver(sat_subset, noisy_d)

    # errors
    err_newton = np.linalg.norm(est_newton - true_position)
    err_linear = np.linalg.norm(est_linear - true_position)

    errors_newton.append(err_newton)
    errors_linear.append(err_linear)


plt.figure()

plt.plot(sat_counts, errors_newton, marker='o', label="Newton")
plt.plot(sat_counts, errors_linear, marker='s', label="Least Squares")

plt.xlabel("Number of satellites")
plt.ylabel("Position error")
plt.title("GPS Error vs Number of Satellites")

plt.legend()
plt.grid()
plt.show()
