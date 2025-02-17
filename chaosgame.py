import numpy as np
import open3d as o3d
import time

#total points to be generated = SETS * AMOUNT_OF_POINTS
SETS = 1
AMOUNT_OF_POINTS = 100

#Simple shapes for chaos game
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
vertices_cube += 0.5


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

vertices_dodecahedron += 0.5


# More advanced shapes
def generate_barnsley_fern_3d(n_points=100000, twist_factor=0.05, depth_factor=0.02):
    # Transformation probabilities
    probabilities = np.array([0.01, 0.85, 0.07, 0.07])
    
    # Affine transformation matrices and translation vectors
    transformations = [
        (np.array([[0, 0], [0, 0.16]]), np.array([0, 0])),  # Stem
        (np.array([[0.85, 0.04], [-0.04, 0.85]]), np.array([0, 1.6])),  # Main transformation
        (np.array([[0.2, -0.26], [0.23, 0.22]]), np.array([0, 1.6])),  # Left leaflet
        (np.array([[-0.15, 0.28], [0.26, 0.24]]), np.array([0, 0.44]))  # Right leaflet
    ]
    
    # Generate points
    points = np.zeros((n_points, 3))  # Now fully 3D

    x, y, z = 0, 0, 0  # Starting point

    for i in range(n_points):
        r = np.random.rand()
        if r < probabilities[0]:
            idx = 0
        elif r < probabilities[:2].sum():
            idx = 1
        elif r < probabilities[:3].sum():
            idx = 2
        else:
            idx = 3

        A, b = transformations[idx]
        x, y = A @ np.array([x, y]) + b
        
        # 3D Effects:
        z = depth_factor * y  # Add depth based on height
        x_rot = x * np.cos(twist_factor * y) - z * np.sin(twist_factor * y)
        z_rot = x * np.sin(twist_factor * y) + z * np.cos(twist_factor * y)
        
        points[i] = [x_rot, y, z_rot]  # Apply rotation

    return points

# Generate the 3D fern
vertices_fern = generate_barnsley_fern_3d(10000, twist_factor=0.05, depth_factor=0.02)


shapes = [vertices_pyramid, vertices_cube, vertices_dodecahedron, vertices_fern]
shape_names = ["Pyramid", "Cube", "Dodecahedron", "Barnsley Fern"]
current_shape = 0  # Start with the first shape
vertices = shapes[current_shape]
fraction = 2

# Initialize Open3D visualization
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(visible=True, width = 600, height = 600)

# Create a dummy PointCloud with one initial point to avoid the empty point warning
initial_point = np.array([[0.5, 0.5, 0.5]])
#initial_point = np.array([[0, 0, 0]])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(initial_point)
vis.add_geometry(pcd)

basis = np.array([[0, 0, 0],[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b_lines = [[0,1], [0,2], [0,3]]
b_line_set = o3d.geometry.LineSet()
b_line_set.points = o3d.utility.Vector3dVector(basis)
b_line_set.lines = o3d.utility.Vector2iVector(b_lines)
b_line_set.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
vis.add_geometry(b_line_set)




def generate_points(vertices, num_points, start_point=None, fraction=2 ):
    """Generates points for the Sierpiński Tetrahedron using the Chaos Game method.
    
    Args:
        num_points (int): Number of points to generate.
        start_point (array-like, optional): Starting point for the fractal generation.
        fraction: How much of the distance to move each iteration (1/fraction)
    
    Returns:
        np.ndarray: Array of generated points.
    """
    #print("given point:", start_point)
    # Use given start point or default to the centroid
    if start_point is None:
        start_point = np.mean(vertices, axis=0)

    #print(len(vertices))
    #initial point
    points = np.zeros((num_points,3))
    vertex = vertices[np.random.randint(0, len(vertices))]
    new_point = (start_point + vertex) / fraction
    points[0] = new_point

    for i in range(1,num_points):
        #print(i)
        # pick random point to go towards
        vertex = vertices[np.random.randint(0, len(vertices))]
        # mid 
        new_point = (points[i-1] + vertex) / fraction
        #print(new_point)
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

def change_fraction(delta):
    global fraction
    fraction += delta
    if fraction == 0:
        print("Fraction was set to 0 -> reset to 1")
        fraction = 1
    print(f"Fraction: {fraction}")
    render_points(vis, vertices, fraction)
    
def adjust_sets(delta):
    global SETS
    SETS += delta
    if SETS == 0:
        print("SETS was set to 0 -> reset to 1")
        SETS = 1
    print(f"SETS (of points): {SETS}")
    render_points(vis, vertices, fraction)
    
def adjust_amount_of_points(delta):
    global AMOUNT_OF_POINTS
    AMOUNT_OF_POINTS += delta
    if AMOUNT_OF_POINTS == 0:
        print("AMOUNT_OF_POINTS was set to 0 -> reset to 1")
        AMOUNT_OF_POINTS = 1
    print(f"AMOUNT_OF_POINTS (per set): {AMOUNT_OF_POINTS}")
    render_points(vis, vertices, fraction)

    
def reset_pointcloud(vis):
    global pcd
    print("Resetting point cloud...")
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))  # Reset to default point
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    #render_points(vis, vertices, fraction)

def switch_shape(delta):
    global current_shape, vertices, shape_index
    current_shape = (current_shape + delta) % len(shapes)
    vertices = shapes[current_shape]
    print(f"Switched to {shape_names[current_shape]}")
    #print(current_shape)
    render_points(vis, vertices, fraction)
    
def render_points(vis, vertices, fraction):
    global current_shape, pcd
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))
    # Generate n sets of x amount of points
    # eg. total points = 10 * 100 points = 1000
    for i in range(SETS):
        #print(" ---------- loop start ")
        prev_points = np.asarray(pcd.points)
        #print("all_points", prev_points)
        #print("latest", prev_points[0])
        new_points = generate_points(vertices, AMOUNT_OF_POINTS, prev_points[0], fraction)
        #new_points = cube(10, prev_points[0])
        #new_points = dodecahedron(1000, prev_points[0])

        new_points = np.vstack((new_points, prev_points))
        #print((new_points))
        
        
        pcd.points = o3d.utility.Vector3dVector(new_points)
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)  # Small delay for visualization effect
        #print(" ---------- loop END ")
        #print()
    print("Total points:", len(np.asarray(pcd.points)))

render_points(vis, vertices, fraction)

vis.register_key_callback(262, lambda vis: switch_shape(1))  # Right Arrow (→) to next shape
vis.register_key_callback(263, lambda vis: switch_shape(-1)) # Left Arrow (←) to previous shape
vis.register_key_callback(265, lambda vis: change_fraction(-1))   # Up Arrow (↑) → Increment step size
vis.register_key_callback(264, lambda vis: change_fraction(1))  # Down Arrow (↓) → Decrement step size
vis.register_key_callback(82, reset_pointcloud) 

vis.register_key_callback(85, lambda vis: adjust_sets(1))   # U → Increment sets (of points)
vis.register_key_callback(74, lambda vis: adjust_sets(-1))  # J → Decrement sets (of points)
vis.register_key_callback(73, lambda vis: adjust_amount_of_points(1))      # I → Increase point amount
vis.register_key_callback(75, lambda vis: adjust_amount_of_points(-1))     # K → Decrement point amount
 
vis.run()
vis.destroy_window()

