import numpy as np
import open3d as o3d
import time

#total points to be generated = SETS * AMOUNT_OF_POINTS
SETS = 10
AMOUNT_OF_POINTS = 100

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

shapes = [vertices_pyramid, vertices_cube, vertices_dodecahedron]
shape_names = ["Pyramid", "Cube", "Dodecahedron"]
current_shape = 0  # Start with the first shape
vertices = shapes[current_shape]
fraction = 2

# Initialize Open3D visualization
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(visible=True, width = 600, height = 600)

# Create a dummy PointCloud with one initial point to avoid the empty point warning
initial_point = np.array([[0, 0, 0]])
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



#print(np.asarray(pcd.points)[-1])

def change_fraction(delta):
    global fraction
    fraction += delta
    if fraction == 0:
        print("fraction was set to 0 -> reset to 1")
        fraction = 1
    print(f"Fraction: {fraction}")
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
    print(len(np.asarray(pcd.points)))

render_points(vis, vertices, fraction)

vis.register_key_callback(262, lambda vis: switch_shape(1))  # Right Arrow (→) to next shape
vis.register_key_callback(263, lambda vis: switch_shape(-1)) # Left Arrow (←) to previous shape
vis.register_key_callback(265, lambda vis: change_fraction(1))   # Up Arrow (↑) → Increment
vis.register_key_callback(264, lambda vis: change_fraction(-1))  # Down Arrow (↓) → Decrement
vis.register_key_callback(82, reset_pointcloud) 

print(len(np.asarray(pcd.points)))
 
vis.run()
vis.destroy_window()

