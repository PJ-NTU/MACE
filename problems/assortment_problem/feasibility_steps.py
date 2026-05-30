"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "placements" not in solution:
        return False, "solution missing 'placements' key"

    placements = solution["placements"]

    if not isinstance(placements, dict):
        return False, f"'placements' must be dict, got {type(placements).__name__}"

    total_piece_counts = [0] * m
    used_stock_types = set()

    # (3)/(4)/(5) per-instance checks
    for stock_instance_id, instance_data in placements.items():
        if not isinstance(instance_data, dict):
            return False, f"stock instance {stock_instance_id} must be a dict"
        if 'stock_type' not in instance_data or 'placements' not in instance_data:
            return False, f"stock instance {stock_instance_id} missing 'stock_type' or 'placements'"

        stock_type = instance_data['stock_type']
        if not isinstance(stock_type, int) or not (1 <= stock_type <= len(stocks)):
            return False, f"stock_type {stock_type} in instance {stock_instance_id} out of range [1, {len(stocks)}]"
        used_stock_types.add(stock_type)

        stock = stocks[stock_type - 1]
        stock_length, stock_width = stock['length'], stock['width']

        placed_rectangles = []

        for placement in instance_data['placements']:
            if not isinstance(placement, dict):
                return False, f"placement in instance {stock_instance_id} must be a dict"

            piece_type = placement.get('piece')
            x = placement.get('x')
            y = placement.get('y')
            orientation = placement.get('orientation')

            if piece_type is None or not (1 <= piece_type <= m):
                return False, f"piece type {piece_type} in instance {stock_instance_id} out of range [1, {m}]"

            piece = pieces[piece_type - 1]

            if orientation == 0:
                p_len, p_wid = piece['length'], piece['width']
            elif orientation == 1:
                p_len, p_wid = piece['width'], piece['length']
            else:
                return False, f"invalid orientation {orientation} for piece {piece_type} in instance {stock_instance_id}"

            if x is None or y is None:
                return False, f"missing x or y for piece {piece_type} in instance {stock_instance_id}"

            if x < 0 or y < 0 or (x + p_len) > stock_length + 1e-6 or (y + p_wid) > stock_width + 1e-6:
                return False, f"piece {piece_type} in instance {stock_instance_id} placed outside stock boundaries"

            rect = (x, y, x + p_len, y + p_wid)
            for other in placed_rectangles:
                if not (rect[2] <= other[0] or rect[0] >= other[2] or rect[3] <= other[1] or rect[1] >= other[3]):
                    return False, f"overlap detected in stock instance {stock_instance_id}"
            placed_rectangles.append(rect)

            total_piece_counts[piece_type - 1] += 1

    # (5) global constraints
    if len(used_stock_types) > 2:
        return False, f"more than 2 distinct stock types used: found {len(used_stock_types)}"

    for idx, piece in enumerate(pieces):
        count = total_piece_counts[idx]
        if count < piece['min'] or count > piece['max']:
            return False, (f"piece type {idx + 1} count {count} violates bounds "
                           f"[{piece['min']}, {piece['max']}]")

    if sum(stocks[st - 1]['length'] * stocks[st - 1]['width'] for st in used_stock_types) == 0 and len(placements) > 0:
        return False, "total stock area is 0, invalid configuration"

    return True, None
'''
