from collections import deque

def solve_warehouse():
    # Representing the warehouse as a two-dimensional grid
    warehouse = [
        list("#####################"),
        list("#S....#..G#.........#"),
        list("#.##.....##########.#"),
        list("#....##..........#..#"),
        list("#.######.###.#.###..#"),
        list("#.......#.........#.#"),
        list("#####################")
    ]
    
    rows, cols = len(warehouse), len(warehouse[0])
    start = goal = None
    
    # Locate Start (S) and Goal (G)
    for r in range(rows):
        for c in range(cols):
            if warehouse[r][c] == 'S': start = (r, c)
            elif warehouse[r][c] == 'G': goal = (r, c)

    # Available actions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue stores: (current_row, current_col, path_taken)
    queue = deque([(start[0], start[1], [])])
    visited = set([start])
    
    while queue:
        r, c, path = queue.popleft()
        
        if (r, c) == goal:
            print(f"Success! Path found in {len(path)} steps.")
            print("Path coordinates:", path)
            return path
            
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds and collision-free spaces
            if 0 <= nr < rows and 0 <= nc < cols:
                if warehouse[nr][nc] != '#' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [(nr, nc)]))
                    
    print("No valid path exists.")
    return None

solve_warehouse()