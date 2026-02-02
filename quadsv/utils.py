import numpy as np


def get_rect_coords(n_rows: int = 32, n_cols: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Generate rectangular grid coordinates with unit spacing.

    Constructs a regular rectangular grid suitable for spatial testing
    on structured data (e.g., image pixels or tissue samples on a regular lattice).

    Parameters
    ----------
    n_rows : int, default 32
        Number of rows in the rectangular grid.
    n_cols : int, default 32
        Number of columns in the rectangular grid.

    Returns
    -------
    coords : np.ndarray
        Grid coordinates of shape (N, 2) where N = n_rows × n_cols.
        Format: [[y₀, x₀], [y₁, x₁], ...] in row-major order.
    grid_dims : tuple of int
        (n_rows, n_cols) - the grid dimensions.

    Notes
    -----
    Coordinates use (row, column) indexing convention where row corresponds
    to the y-axis and column to the x-axis.

    Examples
    --------
    >>> coords, dims = get_rect_coords(n_rows=10, n_cols=10)
    >>> coords.shape
    (100, 2)
    >>> dims
    (10, 10)
    """
    y = np.arange(n_rows)
    x = np.arange(n_cols)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()]), (n_rows, n_cols)


def get_visium_coords(n_rows: int = 78, n_cols: int = 64) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Generate Visium-like hexagonal grid coordinates.

    Constructs coordinates matching the 10x Visium spatial transcriptomics array layout.
    Returns integer indices (array_row, array_col) for the physical array layout where
    spots are arranged in offset rows (alternating by 1 column).

    Parameters
    ----------
    n_rows : int, default 78
        Number of rows in the Visium array (e.g., 78 for full slide, 128 for large tissue).
    n_cols : int, default 64
        Number of spot pairs per row (physical column count is 2 × n_cols or 2 × n_cols + 1).

    Returns
    -------
    coords : np.ndarray
        Array indices of shape (N, 2) where N ≤ n_rows × n_cols.
        Format: [[row₀, col₀], [row₁, col₁], ...].
        Rows are 0-indexed. Columns follow Visium array layout (even/odd offset).
    grid_dims : tuple of int
        (n_rows, n_cols) - the conceptual grid dimensions.

    Notes
    -----
    Visium layout uses offset hexagonal packing:

    - Even rows (0, 2, 4...): column indices 0, 2, 4, 6, ...
    - Odd rows (1, 3, 5...): column indices 1, 3, 5, 7, ...

    This creates a visually offset honeycomb pattern when plotted.
    Use convert_visium_to_physical() to map to physical (y, x) coordinates for distance calculations.

    Examples
    --------
    >>> coords, dims = get_visium_coords(n_rows=78, n_cols=64)
    >>> coords.shape[0]  # Number of spots
    4992
    >>> dims
    (78, 64)
    """
    coords = []
    for r in range(n_rows):
        # Determine the starting column index based on row parity
        # Even rows (0, 2...): cols are 0, 2, 4...
        # Odd rows (1, 3...): cols are 1, 3, 5...
        start_col = 0 if r % 2 == 0 else 1

        for i in range(n_cols):
            c = start_col + (i * 2)
            coords.append([r, c])

    return np.array(coords), (n_rows, n_cols)


def convert_visium_to_physical(coords: np.ndarray) -> np.ndarray:
    """
    Convert integer Visium indices to physical hexagonal coordinates.

    Maps array row/column indices to physical (y, x) coordinates in a hexagonal grid
    with equilateral triangle geometry. Suitable for distance-based kernel construction
    and visualization.

    Parameters
    ----------
    coords : np.ndarray
        Visium array indices of shape (N, 2) in format [[row, col], ...].
        Rows and columns are integers as returned by get_visium_coords().

    Returns
    -------
    phys_coords : np.ndarray
        Physical coordinates of shape (N, 2) in format [[y, x], ...].

    Notes
    -----
    Visium uses hexagonal packing with offset rows. The conversion assumes:

    - Horizontal neighbor spacing: Δcol = 2 → Δx = 1
    - Vertical neighbor spacing: Δrow = 1 → Δy = √3/2 (hexagonal geometry)

    Formulas:

    .. math::
       x = \\text{col} / 2 \\quad
       y = \\text{row} \\cdot \\frac{\\sqrt{3}}{2}

    This ensures that nearest neighbors form equilateral triangles in the physical space.

    Examples
    --------
    >>> coords = np.array([[0, 0], [0, 2], [1, 1]])  # Visium indices
    >>> phys_coords = convert_visium_to_physical(coords)
    >>> phys_coords
    array([[0.       , 0.       ],
           [0.       , 1.       ],
           [0.8660254, 0.5      ]])
    """
    # Separate row and col
    rows = coords[:, 0]
    cols = coords[:, 1]

    # Calculate physical coordinates
    # We use sqrt(3)/2 for Y spacing to ensure equilateral triangles
    phys_y = rows * np.sqrt(3) / 2.0
    phys_x = cols * 0.5

    return np.column_stack([phys_y, phys_x])


def compute_torus_distance_matrix(
    phys_coords: np.ndarray, domain_dims: tuple[float, float]
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances on a torus (with periodic boundary conditions).

    Calculates distances between all pairs of points assuming the spatial domain
    is periodic (wrapping at boundaries). Useful for analyzing spatial patterns in
    periodic simulations or tissue with natural periodicity.

    Parameters
    ----------
    phys_coords : np.ndarray
        Physical coordinates of shape (N, 2) in format [[y, x], ...].
    domain_dims : tuple of float
        Physical dimensions of the periodic domain: (height, width).

    Returns
    -------
    dist_matrix : np.ndarray
        Pairwise distance matrix of shape (N, N) where entry [i, j] is the
        shortest Euclidean distance between points i and j on the torus.

    Notes
    -----
    On a torus, the distance between two points is computed as the minimum of:
    1. Direct Euclidean distance
    2. Euclidean distance after wrapping through periodic boundaries

    For each dimension, the wrapped distance is

    .. math::
       d_{wrapped} = \\min(|\\Delta p|, D - |\\Delta p|),

    where D is the domain extent in that dimension.

    Examples
    --------
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [9.9, 0.0]])
    >>> domain = (10.0, 10.0)
    >>> dists = compute_torus_distance_matrix(coords, domain)
    >>> dists[0, 2]  # Distance from (0,0) to (9.9,0) on [0,10)×[0,10) torus
    0.1  # Wraps around
    """
    domain_h, domain_w = domain_dims

    # Compute absolute differences (broadcasting)
    # Shape: (N, N)
    diff_y = np.abs(phys_coords[:, None, 0] - phys_coords[None, :, 0])
    diff_x = np.abs(phys_coords[:, None, 1] - phys_coords[None, :, 1])

    # Apply Torus wrapping (minimum of direct distance vs wrapped distance)
    wrapped_dy = np.minimum(diff_y, domain_h - diff_y)
    wrapped_dx = np.minimum(diff_x, domain_w - diff_x)

    return np.sqrt(wrapped_dy**2 + wrapped_dx**2)
