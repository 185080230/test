import numpy as np
from scipy.interpolate import griddata
from wind_model import calculate_wind_speed_model

def Cubic_spline_interpolation(coords, data, grid_x, grid_y):
    grid_coords = np.meshgrid(grid_x, grid_y)
    grid_coords = np.column_stack([grid_coords[0].ravel(), grid_coords[1].ravel()])
    interpolated_data = griddata(coords, data, grid_coords, method='cubic')
    interpolated_data = interpolated_data.reshape((len(grid_y), len(grid_x)))
    return interpolated_data
def calculate_wind_speed(x, y):
    return calculate_wind_speed_model(x, y)
def exponential_hdf(z, z_prime, gamma):
    return np.exp(-np.linalg.norm(np.array(z) - np.array(z_prime)) / gamma)
def non_local_diffusion(initial_concentration, grid_x, grid_y, wind_speed, dt, dx, dy, num_iterations, gamma):
    concentration = initial_concentration.copy()
    for _ in range(num_iterations):
        non_local_term_x = np.zeros_like(concentration)
        non_local_term_y = np.zeros_like(concentration)
        for i in range(concentration.shape[0]):
            for j in range(concentration.shape[1]):
                for k in range(concentration.shape[0]):
                    for l in range(concentration.shape[1]):
                        weight = exponential_hdf((grid_x[i], grid_y[j]), (grid_x[k], grid_y[l]), gamma)
                        non_local_term_x[i, j] += concentration[k, l] * weight * wind_speed[k, l] * np.cos(np.radians(wind_direction))
                        non_local_term_y[i, j] += concentration[k, l] * weight * wind_speed[k, l] * np.sin(np.radians(wind_direction))
        concentration_x = np.roll(concentration, 1, axis=0) + np.roll(concentration, -1, axis=0)
        concentration_y = np.roll(concentration, 1, axis=1) + np.roll(concentration, -1, axis=1)
        concentration += dt * ((non_local_term_x + non_local_term_y) / (dx**2 + dy**2)) + dt * ((concentration_x + concentration_y - 2 * concentration) / (dx**2 + dy**2))
    return concentration
def main():
    global wind_direction
    wind_direction = 45
    sensor_coords = np.array([[0, 0], [40, 40], [80, 0], [80, 80], [0, 80]])
    concentration_data = np.array([119, 153, 108, 128, 122])
    grid_x = np.linspace(0, 80, 28)
    grid_y = np.linspace(0, 80, 28)
    interpolated_data = Cubic_spline_interpolation(sensor_coords, concentration_data, grid_x, grid_y)
    wind_speed = np.zeros((len(grid_y), len(grid_x)))
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            wind_speed[j, i] = calculate_wind_speed(grid_x[i], grid_y[j])
    dt = 0.01
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    num_iterations = 50
    gamma = 10
    final_concentration = non_local_diffusion(interpolated_data, grid_x, grid_y, wind_speed, dt, dx, dy, num_iterations, gamma)
    max_index = np.unravel_index(np.argmax(final_concentration), final_concentration.shape)
    max_coords = (grid_x[max_index[1]], grid_y[max_index[0]])
    print("最高浓度坐标：", max_coords)
if __name__ == "__main__":
    main()
