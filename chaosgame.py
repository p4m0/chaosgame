import numpy as np
import open3d as o3d
import time

# Define the function to compute the 2D Mandelbrot set
def mandelbrot_2d(resolution, max_iter, bounds):
    x_min, x_max, y_min, y_max = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)

    # Create a grid of points in 2D space
    X, Y = np.meshgrid(x, y)
    
    # Initialize c and z arrays
    c = X + Y * 1j
    z = np.zeros_like(c, dtype=np.complex128)

    # Initialize a boolean array for points in the Mandelbrot set
    mandelbrot_set = np.ones(c.shape, dtype=bool)

    # Iterate to check if points belong to the Mandelbrot set
    for _ in range(max_iter):
        z = z**2 + c
        diverged = np.abs(z) > 2
        mandelbrot_set &= ~diverged
        z[diverged] = 0

    return np.argwhere(mandelbrot_set)  # Return indices of points in the set


def generate_points(vertices, num_points, start_point=None, fraction=2 ):
    """Generates points for the Sierpiński Tetrahedron using the Chaos Game method.
    
    Args:
        num_points (int): Number of points to generate.
        start_point (array-like, optional): Starting point for the fractal generation.
        fraction: How much of the distance to move each iteration (1/fraction)
    
    Returns:
        np.ndarray: Array of generated points.
    """
    print("given point:", start_point)
    # Use given start point or default to the centroid
    if start_point is None:
        start_point = np.mean(vertices, axis=0)

    #print(len(vertices))
    points = np.zeros((num_points,3))

    for i in range(num_points):
        # pick random point to go towards
        vertex = vertices[np.random.randint(0, len(vertices))]
        # mid 
        #points[i] = 0,0,0
        if i == 0:
            new_point = (start_point + vertex) / fraction
        else:
            new_point = (points[i-1] + vertex) / fraction
        print(new_point)
        points[i] = new_point
    #print("NEW POINTS", points)

    return np.array(points)

def point_check(pcdpoints):
    # Convert array to tuples for easy comparison
    points_tuples = [tuple(row) for row in np.asarray(pcdpoints) ]

    # Count occurrences
    unique_points, counts = np.unique(points_tuples, axis=0, return_counts=True)

    # Print results
    for point, count in zip(unique_points, counts):
        print(f"Point {point} appears {count} times")

vertices_pyramid = np.array([
    [0, 0, 0],        
    [1, 0, 0],        
    [0.5, np.sqrt(3)/2, 0],  
    [0.5, np.sqrt(3)/6, np.sqrt(2/3)]  
])

vertices_cube = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [-0.5,  0.5,  0.5],
    [-0.5,  0.5, -0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5]
])

phi = (1 + np.sqrt(5)) / 2  

# Dodecahedron vertices
vertices_dodecahedron = np.array([
    # (±1, ±1, ±1)
    [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1],
    
    # (0, ±1/φ, ±φ)
    [0, -1/phi, -phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, 1/phi, phi],
    
    # (±1/φ, ±φ, 0)
    [-1/phi, -phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [1/phi, phi, 0],
    
    # (±φ, 0, ±1/φ)
    [-phi, 0, -1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [phi, 0, 1/phi]
])

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True, width = 600, height = 600)

# Create a dummy PointCloud with one initial point to avoid the empty point warning
initial_point = np.array([[0.5, 0.5, 0.5]])
#initial_point = np.array([[0, 0, 0]])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(initial_point)
vis.add_geometry(pcd)



#basis = np.array([[0, 0, 0],[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#b_lines = [[0,1], [0,2], [0,3]]
#b_line_set = o3d.geometry.LineSet()
#b_line_set.points = o3d.utility.Vector3dVector(basis)
#b_line_set.lines = o3d.utility.Vector2iVector(b_lines)
#b_line_set.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
#vis.add_geometry(b_line_set)
print(np.asarray(pcd.points)[-1])

animate = False
if animate == False:
# Generate points and update in real-time
    for i in range(10):
        #print(" ---------- loop start ")
        prev_points = np.asarray(pcd.points)
        #print("all_points", prev_points)
        #print("latest", prev_points[0])
        new_points = generate_points(vertices_dodecahedron, 100, prev_points[0], fraction=4)
        #new_points = cube(10, prev_points[0])
        #new_points = dodecahedron(1000, prev_points[0])

        new_points = np.vstack((new_points, prev_points))
        #print((new_points))
        
        pcd.points = o3d.utility.Vector3dVector(new_points)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # Small delay for visualization effect
        #print(" ---------- loop END ")
        #print()
else:
    print("a")


    
print(len(np.asarray(pcd.points)))
 
vis.run()
vis.destroy_window()

"""
t_points = sierpinski_tetrahedron(1000)
print(t_points)
pcd.points = o3d.utility.Vector3dVector(t_points)

vis.add_geometry(pcd)
vis.update_renderer()
vis.poll_events()
#o3d.visualization.draw_geometries([pcd])
# Keep window open
vis.run()
vis.destroy_window()"""


"""
# Parameters for the Mandelbrot set
resolution = 500
max_iter = 50
bounds = (-2, 1, -1.5, 1.5)

# Generate the Mandelbrot set in 2D
mandelbrot_points = mandelbrot_2d(resolution, max_iter, bounds)

# Map indices back to 2D coordinates
x = np.linspace(bounds[0], bounds[1], resolution)
y = np.linspace(bounds[2], bounds[3], resolution)
points_2d = np.array([[x[i], y[j]] for i, j in mandelbrot_points])

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.hstack((points_2d, np.zeros((points_2d.shape[0], 1)))))

# Visualize the Mandelbrot set
o3d.visualization.draw_geometries([pcd])"""
