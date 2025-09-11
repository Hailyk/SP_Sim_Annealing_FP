import random
import math
import time
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Data Structures ---

class Block:
    """Represents a single block with width, height, and name."""
    def __init__(self, name, width, height, is_fixed=False, x=0, y=0):
        self.name = name
        self.width = width
        self.height = height
        self.is_fixed = is_fixed
        # For fixed blocks, x and y store their target position
        # For all blocks, they will store the final calculated position
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Block({self.name}, w={self.width}, h={self.height}, fixed={self.is_fixed})"

# --- File Parsing ---

def parse_blocks(filename):
    """
    Parses a .blocks file to create a dictionary of Block objects.
    Supports optional fixed coordinates: <name> <width> <height> [x y]
    """
    blocks = {}
    with open(filename, 'r') as f:
        # Skip header lines until we find NumBlocks
        for line in f:
            if line.strip().startswith("NumBlocks"):
                break

        for line in f:
            if line.strip(): # Ignore empty lines
                parts = line.split()
                name = parts[0]
                width, height = int(parts[1]), int(parts[2])

                if len(parts) == 5:
                    x, y = int(parts[3]), int(parts[4])
                    blocks[name] = Block(name, width, height, is_fixed=True, x=x, y=y)
                else:
                    blocks[name] = Block(name, width, height)
    return blocks

def parse_nets(filename):
    """Parses a .nets file to create a list of net connections."""
    nets = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith("NumNets"):
                break

        for line in f:
            line = line.strip()
            if line.startswith("Net"):
                parts = line.split(":")
                connected_blocks = parts[1].strip().split()
                nets.append(connected_blocks)
    return nets

# --- Core Sequence Pair Logic ---

def evaluate_floorplan(seq_plus, seq_minus, blocks, nets):
    """
    Calculates area and wirelength from a sequence pair, allowing empty space for fixed blocks.

    Returns:
        tuple: (total_area, total_hpwl). Returns (inf, inf) for illegal layouts.
    """
    block_names = list(blocks.keys())
    n = len(block_names)

    pos_plus = {name: i for i, name in enumerate(seq_plus)}
    pos_minus = {name: i for i, name in enumerate(seq_minus)}

    # --- 1. Build Constraint Graphs ---
    hcg = {name: [] for name in block_names}
    vcg = {name: [] for name in block_names}

    for i in range(n):
        for j in range(i + 1, n):
            b_i_name, b_j_name = block_names[i], block_names[j]
            if pos_plus[b_i_name] < pos_plus[b_j_name] and pos_minus[b_i_name] < pos_minus[b_j_name]:
                hcg[b_i_name].append(b_j_name)
            elif pos_plus[b_i_name] > pos_plus[b_j_name] and pos_minus[b_i_name] > pos_minus[b_j_name]:
                hcg[b_j_name].append(b_i_name)

            if pos_plus[b_i_name] < pos_plus[b_j_name] and pos_minus[b_i_name] > pos_minus[b_j_name]:
                vcg[b_i_name].append(b_j_name)
            elif pos_plus[b_i_name] > pos_plus[b_j_name] and pos_minus[b_i_name] < pos_minus[b_j_name]:
                vcg[b_j_name].append(b_i_name)

    # --- 2. Calculate Final Coordinates with Slack ---
    x_coords = _calculate_coords(hcg, {name: b.width for name, b in blocks.items()}, blocks, is_x_coords=True)
    y_coords = _calculate_coords(vcg, {name: b.height for name, b in blocks.items()}, blocks, is_x_coords=False)

    # If either coordinate calculation failed, the layout is illegal
    if x_coords is None or y_coords is None:
        return (sys.float_info.max, sys.float_info.max)

    # --- 3. Update Block objects and Calculate Area ---
    for name, b in blocks.items():
        b.x, b.y = x_coords[name], y_coords[name]

    total_width = max(b.x + b.width for b in blocks.values())
    total_height = max(b.y + b.height for b in blocks.values())
    total_area = total_width * total_height

    # --- 4. Calculate Total Wirelength (HPWL) ---
    total_hpwl = 0
    for net in nets:
        if not net: continue
        block_A_center_x = blocks[net[0]].x + blocks[net[0]].width / 2
        block_A_center_y = blocks[net[0]].y + blocks[net[0]].height / 2

        block_B_center_x = blocks[net[1]].x + blocks[net[1]].width / 2
        block_B_center_y = blocks[net[1]].y + blocks[net[1]].height / 2

        hpwl = abs(block_A_center_x - block_B_center_x) + abs(block_A_center_y - block_B_center_y)
        total_hpwl += hpwl

    return total_area, total_hpwl


def _calculate_coords(graph, weights, blocks, is_x_coords):
    """
    Calculates coordinates for a given dimension (x or y) using a topological sort.
    It respects fixed block locations and allows for empty space.
    Returns a dictionary of coordinates or None if the layout is illegal.
    """
    # 1. Get topological sort order
    in_degree = {name: 0 for name in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = [name for name in graph if in_degree[name] == 0]
    topo_order = []
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != len(graph):
        raise Exception("Cycle detected in constraint graph")

    # 2. Calculate coordinates by propagating constraints
    coords = {name: 0 for name in graph}
    for u_name in topo_order:
        u_block = blocks[u_name]

        # If block is fixed, check for legality and enforce its position
        if u_block.is_fixed:
            target_coord = u_block.x if is_x_coords else u_block.y
            # If relative constraints already pushed the block past its fixed location, it's illegal
            if coords[u_name] > target_coord:
                return None
            # Lock the coordinate to its fixed value, creating slack if necessary
            coords[u_name] = target_coord

        # Propagate the calculated coordinate to successors
        for v_name in graph[u_name]:
            coords[v_name] = max(coords[v_name], coords[u_name] + weights[u_name])

    return coords

# --- Simulated Annealing ---

def generate_neighbor(seq_plus, seq_minus):
    """Generates a neighboring solution by making a small random change."""
    sp, sm = list(seq_plus), list(seq_minus)
    n = len(sp)
    move = random.randint(1, 2)

    if move == 1: # Swap two blocks in one sequence
        seq_to_modify = random.choice([sp, sm])
        i, j = random.sample(range(n), 2)
        seq_to_modify[i], seq_to_modify[j] = seq_to_modify[j], seq_to_modify[i]
    elif move == 2: # Swap two blocks in both sequences
        i, j = random.sample(range(n), 2)
        sp[i], sp[j] = sp[j], sp[i]
        sm[i], sm[j] = sm[j], sm[i]

    return tuple(sp), tuple(sm)

def simulated_annealing(blocks, nets, alpha):
    """
    Performs the simulated annealing optimization.
    """
    block_names = list(blocks.keys())

    # --- 1. Initial State ---
    current_sp = random.sample(block_names, len(block_names))
    current_sm = random.sample(block_names, len(block_names))

    initial_area, initial_wl = evaluate_floorplan(current_sp, current_sm, blocks, nets)
    # Ensure the initial state is legal (can satisfy fixed constraints)
    attempts = 0
    while initial_area == sys.float_info.max:
        print("Finding a valid initial configuration. attempt:", attempts, end='\r')
        current_sp = random.sample(block_names, len(block_names))
        current_sm = random.sample(block_names, len(block_names))
        initial_area, initial_wl = evaluate_floorplan(current_sp, current_sm, blocks, nets)
        attempts += 1



    print(f"Initial Area: {initial_area:.2f}, Initial Wirelength: {initial_wl:.2f}      ")
    visualize_floorplan(blocks, title="Initial Floorplan")

    current_area, current_wl = initial_area, initial_wl
    current_cost = alpha * (current_area / initial_area) + (1 - alpha) * (current_wl / initial_wl) if initial_wl > 0 else alpha * (current_area / initial_area)

    best_sp, best_sm = current_sp, current_sm
    best_area, best_wl = current_area, current_wl
    best_cost = current_cost

    # --- 2. Annealing Schedule ---
    T_initial, T_final = 100000.0, 100
    cooling_rate = 0.99
    steps_per_temp = 100 * len(block_names)
    T = T_initial
    start_time = time.time()

    while T > T_final:
        for _ in range(steps_per_temp):
            # --- 3. Generate and Evaluate Neighbor ---
            new_sp, new_sm = generate_neighbor(current_sp, current_sm)
            new_area, new_wl = evaluate_floorplan(new_sp, new_sm, blocks, nets)

            if new_area == sys.float_info.max: continue # Skip illegal moves

            norm_area = new_area / initial_area
            norm_wl = new_wl / initial_wl if initial_wl > 0 else 0
            new_cost = alpha * norm_area + (1 - alpha) * norm_wl

            # --- 4. Acceptance Criteria ---
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_sp, current_sm = new_sp, new_sm
                current_area, current_wl = new_area, new_wl
                current_cost = new_cost
                if current_cost < best_cost:
                    best_sp, best_sm, best_area, best_wl, best_cost = current_sp, current_sm, current_area, current_wl, current_cost

        # --- 5. Cool Down ---
        T *= cooling_rate
        print(f"Temp: {T:7.2f}, Current Area: {current_area:10.0f}, Current WL: {current_wl:10.0f}, Best Area: {best_area:10.0f}, Best WL: {best_wl:10.0f}", end='\r')

    end_time = time.time()
    print("\n" + "="*50)
    print("Optimization Finished")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Final Best Area: {best_area:.2f}")
    print(f"Final Best Wirelength: {best_wl:.2f}")

    # Recalculate final state to get final dimensions and positions
    evaluate_floorplan(best_sp, best_sm, blocks, nets)
    final_w = max(b.x + b.width for b in blocks.values())
    final_h = max(b.y + b.height for b in blocks.values())
    print(f"Final Dimensions: {final_w:.2f} x {final_h:.2f}")
    print("Final Block Positions:")
    for name in sorted(blocks.keys()):
        b = blocks[name]
        print(f"  - {b.name:<5}: (x={b.x:<5.0f}, y={b.y:<5.0f})")
    print("="*50)

    return blocks


def visualize_floorplan(blocks, title="Final Floorplan", save_filename=None):
    """
    Generates a visual plot of the floorplan using matplotlib.

    Args:
        blocks (dict): A dictionary of Block objects with final x, y coordinates.
        title (str): The title for the plot.
        save_filename (str, optional): If provided, saves the plot to this file.
    """
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Find the total width and height to set the plot limits
    if not blocks:
        print("Warning: No blocks to visualize.")
        return

    total_width = max(b.x + b.width for b in blocks.values())
    total_height = max(b.y + b.height for b in blocks.values())

    # Set plot limits and aspect ratio
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.set_aspect('equal', adjustable='box')

    # Use a colormap to get distinct colors for blocks
    # Using 'viridis' colormap and getting N colors from it
    colors = plt.cm.get_cmap('viridis', len(blocks))

    # Iterate through each block and draw it
    for i, (name, block) in enumerate(blocks.items()):
        # Define rectangle properties
        face_color = colors(i)
        edge_color = 'red' if block.is_fixed else 'black'
        line_width = 2.0 if block.is_fixed else 1.0

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (block.x, block.y),      # (x,y) bottom-left corner
            block.width,             # width
            block.height,            # height
            linewidth=line_width,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=0.75               # transparency
        )

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add the block name as a text label in the center
        center_x = block.x + block.width / 2
        center_y = block.y + block.height / 2
        ax.text(center_x, center_y, block.name,
                ha='center', va='center', color='white', weight='bold')

    # Configure plot appearance
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the file if a name is provided
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"\nFloorplan visualization saved to '{save_filename}'")

    # Display the plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Floorplanner using Simulated Annealing with Sequence Pairs.")
    parser.add_argument("blocks_file", help="Path to the .blocks input file.")
    parser.add_argument("nets_file", help="Path to the .nets input file.")
    parser.add_argument("alpha", type=float, help="Weight for area vs. wirelength (0.0 to 1.0). 1.0 for area only.")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("Error: Alpha must be between 0.0 and 1.0.")
        sys.exit(1)

    try:
        blocks = parse_blocks(args.blocks_file)
        nets = parse_nets(args.nets_file)
        blocks = simulated_annealing(blocks, nets, args.alpha)
        visualize_floorplan(blocks)
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
