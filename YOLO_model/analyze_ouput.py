def parse_yolo_output(yolo_output, image_width, image_height):
    """
    Parse YOLO output to extract features and calculate center points for tree parts.

    Args:
        yolo_output (list): List of detected objects with labels, bounding boxes, and confidence scores.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.

    Returns:
        dict: Parsed tree features including bounding boxes and center points.
    """
    features = {}

    for detection in yolo_output:
        label = detection["label"]  # e.g., "trunk", "canopy", "root"
        bbox = detection["bbox"]  # Bounding box [x_min, y_min, x_max, y_max]
        confidence = detection["confidence"]

        # Calculate the center of the bounding box
        center_x = (bbox[0] + bbox[2]) / 2 / image_width
        center_y = (bbox[1] + bbox[3]) / 2 / image_height

        features[label] = {
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "confidence": confidence
        }

    return features
def analyze_tree_proportions(features, image_width, image_height, indicator):
    """
    Analyze overall tree proportions in relation to the image size.

    Args:
        features (dict): Parsed tree features.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.
        indicator (dict): Indicator thresholds and interpretations.

    Returns:
        list: Insights related to tree proportions.
    """
    analysis = []

    # Validate required features
    if "canopy" not in features or "trunk" not in features:
        analysis.append({
            "trait": "חלקי עץ חסרים",
            "description": "העץ חסר רכיבים חשובים כמו חופה או גזע, מה שמקשה על פרשנות הפרופורציות.",
            "indicator": "חופה או גזע חסרים בציור"
        })
        return analysis

    # Calculate total tree height and width
    canopy_bbox = features['canopy']['bbox']
    trunk_bbox = features['trunk']['bbox']

    tree_width = max(
        canopy_bbox[2] - canopy_bbox[0],
        trunk_bbox[2] - trunk_bbox[0]
    )
    tree_height = max(
        canopy_bbox[3] - canopy_bbox[1],
        trunk_bbox[3] - trunk_bbox[1]
    )

    # Include root height if present
    if "root" in features:
        root_bbox = features['root']['bbox']
        root_height = root_bbox[3] - root_bbox[1]
        tree_height = max(tree_height, root_height)

    total_tree_area = tree_width * tree_height
    total_image_area = image_width * image_height
    tree_area_ratio = total_tree_area / total_image_area

    # Add indicators for tree size relative to image size
    thresholds = indicator['tree_proportions']['relative_to_image']
    if tree_area_ratio < thresholds["small_threshold"]:
        analysis.append({
            "trait": "ביטחון עצמי",
            "description": thresholds["interpretation"]["small"],
            "indicator": "הפרופורציה בין העץ המצויר לשאר הדף קטנה יחסית"
        })
    elif tree_area_ratio > thresholds["large_threshold"]:
        analysis.append({
            "trait": "ביטחון עצמי",
            "description": thresholds["interpretation"]["large"],
            "indicator": "הפרופורציה בין העץ המצויר לשאר הדף גדולה יחסית "
        })

    # Trunk-to-canopy ratio analysis
    trunk_height = trunk_bbox[3] - trunk_bbox[1]
    canopy_height = canopy_bbox[3] - canopy_bbox[1]
    trunk_canopy_ratio = trunk_height / canopy_height

    trunk_thresholds = indicator['tree_proportions']['features_relative_to_each_other']['trunk_canopy_ratio']
    if trunk_canopy_ratio < trunk_thresholds["thin_trunk_threshold"]:
        analysis.append({
            "trait": "אופי גמיש",
            "description": trunk_thresholds["interpretation"]["thin_trunk"],
            "indicator": "היחס בין הגזע לחופה מעיד על גמישות"
        })
    elif trunk_canopy_ratio > trunk_thresholds["thick_trunk_threshold"]:
        analysis.append({
            "trait": "יציבות וביטחון",
            "description": trunk_thresholds["interpretation"]["thick_trunk"],
            "indicator": "היחס בין הגזע לחופה מעיד על יציבות"
        })
    # Root presence analysis
    root_thresholds = indicator['tree_proportions']['features_relative_to_each_other']['root_presence']
    if "root" not in features:
        analysis.append({
            "trait": "שלב חיפוש",
            "description": root_thresholds["interpretation"]["absent"],
            "indicator": "השורשים חסרים בציור"
        })
    else:
        analysis.append({
            "trait": "חיבור לערכים",
            "description": root_thresholds["interpretation"]["present"],
            "indicator": "השורשים נוכחים בציור"
        })

    return analysis



def analyze_feature_proportions(features, indicator):
    """
    Analyze the proportions of tree features relative to each other.

    Args:
        features (dict): Parsed tree features.
        indicator (dict): Indicator thresholds and interpretations.

    Returns:
        list: Insights related to the relationship between features (e.g., trunk vs. canopy).
    """
    analysis = []

    if "trunk" in features and "canopy" in features:
        trunk_bbox = features['trunk']['bbox']
        canopy_bbox = features['canopy']['bbox']

        trunk_width = trunk_bbox[2] - trunk_bbox[0]
        canopy_width = canopy_bbox[2] - canopy_bbox[0]

        # Check trunk size relative to canopy
        thresholds = indicator['tree_proportions']['features_relative_to_each_other']['trunk_canopy_ratio']
        if trunk_width < canopy_width * thresholds["thin_trunk_threshold"]:
            analysis.append({
                "trait": "כוח פנימי",
                "description": thresholds["interpretation"]["thin_trunk"],
                "indicator": "הגזע דק יחסית לחופה"
            })
        elif trunk_width > canopy_width * thresholds["thick_trunk_threshold"]:
            analysis.append({
                "trait": "מיקוד יציב",
                "description": thresholds["interpretation"]["thick_trunk"],
                "indicator": "הגזע רחב יחסית לחופה"
            })

    # Root presence analysis
    root_thresholds = indicator['tree_proportions']['features_relative_to_each_other']['root_presence']
    if "root" in features:
        analysis.append({
            "trait": "נוכחות שורשים",
            "description": root_thresholds["interpretation"]["present"],
            "indicator": "נוכחות שורשים בציור"
        })
    else:
        analysis.append({
            "trait": "חסר שורשים",
            "description": root_thresholds["interpretation"]["absent"],
            "indicator": "חסרים שורשים בציור"
        })

    return analysis



def analyze_tree_location(features, image_width, image_height, indicator):
    """
    Analyze tree's position on the page in relation to self-image and social relations.

    Args:
        features (dict): Parsed tree features.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.
        indicator (dict): Indicator thresholds and interpretations.

    Returns:
        list: Insights based on tree positioning.
    """
    analysis = []

    if "canopy" not in features:
        analysis.append({
            "trait": "חסר חופה",
            "description": "לא ניתן לנתח את מיקום העץ ללא חופה.",
            "indicator": "missing_canopy"
        })
        return analysis

    canopy_center_x = features['canopy']['center_x']
    canopy_center_y = features['canopy']['center_y']

    horizontal_thresholds = indicator['tree_location']['horizontal_position']
    vertical_thresholds = indicator['tree_location']['vertical_position']

    # Analyze horizontal position
    if canopy_center_x < horizontal_thresholds["left_threshold"]:
        analysis.append({
            "trait": "התמקדות בעבר",
            "description": horizontal_thresholds["interpretation"]["left"],
            "indicator": f"canopy_center_x < {horizontal_thresholds['left_threshold']}"
        })
    elif canopy_center_x > horizontal_thresholds["right_threshold"]:
        analysis.append({
            "trait": "התמקדות בעתיד",
            "description": horizontal_thresholds["interpretation"]["right"],
            "indicator": f"canopy_center_x > {horizontal_thresholds['right_threshold']}"
        })
    elif (horizontal_thresholds["center_1_threshold"] <= canopy_center_x and
                horizontal_thresholds["center_2_threshold"] >= canopy_center_x):
        analysis.append({
            "trait": "תפיסה מאוזנת",
            "description": horizontal_thresholds["interpretation"]["center"],
            "indicator": f"{horizontal_thresholds['center_1_threshold']} <= canopy_center_x <= {horizontal_thresholds['center_2_threshold']}"
        })

    # Analyze vertical position
    if canopy_center_y < vertical_thresholds["low_threshold"]:
        analysis.append({
            "trait": "מחובר לקרקע",
            "description": vertical_thresholds["interpretation"]["low"],
            "indicator": f"canopy_center_y < {vertical_thresholds['low_threshold']}"
        })
    elif canopy_center_y > vertical_thresholds["high_threshold"]:
        analysis.append({
            "trait": "שאיפות גבוהות",
            "description": vertical_thresholds["interpretation"]["high"],
            "indicator": f"canopy_center_y > {vertical_thresholds['high_threshold']}"
        })
    elif (vertical_thresholds["center_1_threshold"] <= canopy_center_y and
                vertical_thresholds["center_2_threshold"] >= canopy_center_y):
        analysis.append({
            "trait": "יציבות רגשית",
            "description": vertical_thresholds["interpretation"]["center"],
            "indicator": f"{vertical_thresholds['center_1_threshold']} <= canopy_center_y <= {vertical_thresholds['center_2_threshold']}"
        })

    return analysis



def analyze_tree_shapes(features, indicator):
    """
    Analyze the shapes of tree parts like canopy and trunk.

    Args:
        features (dict): Parsed tree features.
        indicator (dict): Indicator thresholds and interpretations.

    Returns:
        list: Insights related to shapes.
    """
    analysis = []

    # Analyze canopy shape
    if "canopy" in features and "shape" in features['canopy']:
        shape = features['canopy']['shape']
        if shape in indicator['canopy']['shape']:
            analysis.append({
                "trait": "צורת כתר",
                "description": indicator['canopy']['shape'][shape],
                "indicator": f"canopy_shape = {shape}"
            })
        # else:
        #     analysis.append({
        #         "trait": "צורת כתר לא מוכרת",
        #         "description": "צורת הכתר לא זוהתה במאגר הנתונים.",
        #         "indicator": f"canopy_shape = {shape} (לא מוכר)"
        #     })
    # else:
        # analysis.append({
        #     "trait": "חסר כתר",
        #     "description": "לא ניתן לנתח את צורת הכתר כי הוא חסר.",
        #     "indicator": "missing_canopy_shape"
        # })

    # Analyze trunk shape
    if "trunk" in features and "shape" in features['trunk']:
        shape = features['trunk']['shape']
        if shape in indicator['trunk']['shape']:
            analysis.append({
                "trait": "צורת גזע",
                "description": indicator['trunk']['shape'][shape],
                "indicator": f"trunk_shape = {shape}"
            })
        # else:
        #     analysis.append({
        #         "trait": "צורת גזע לא מוכרת",
        #         "description": "צורת הגזע לא זוהתה במאגר הנתונים.",
        #         "indicator": f"trunk_shape = {shape} (לא מוכר)"
        #     })
    # else:
        # analysis.append({
        #     "trait": "חסר גזע",
        #     "description": "לא ניתן לנתח את צורת הגזע כי הוא חסר.",
        #     "indicator": "missing_trunk_shape"
        # })

    # Analyze root depth
    if "root" in features and "depth" in features['root']:
        root_depth = features['root']['depth']
        thresholds = indicator['roots']['depth']
        if root_depth < thresholds["shallow_threshold"]:
            analysis.append({
                "trait": "שורשים רדודים",
                "description": thresholds["interpretation"]["shallow"],
                "indicator": f"root_depth < {thresholds['shallow_threshold']}"
            })
        elif root_depth > thresholds["deep_threshold"]:
            analysis.append({
                "trait": "שורשים עמוקים",
                "description": thresholds["interpretation"]["deep"],
                "indicator": f"root_depth > {thresholds['deep_threshold']}"
            })
        else:
            analysis.append({
                "trait": "שורשים בעומק בינוני",
                "description": "העומק של השורשים מתאים לאמצע הסקאלה.",
                "indicator": f"shallow_threshold <= root_depth <= deep_threshold"
            })
    # else:
    #     analysis.append({
    #         "trait": "חסרי שורשים",
    #         "description": "לא ניתן לנתח את עומק השורשים כי הם חסרים.",
    #         "indicator": "missing_root_depth"
    #     })

    return analysis



def analyze_tree_size_individual(features, image_width, image_height, indicator):
    analysis = []

    # if "canopy" not in features:
    #     analysis.append({
    #         "trait": "חסר חופה",
    #         "description": "לא ניתן לנתח את גודל החופה כי היא חסרה.",
    #         "indicator": "missing_canopy"
    #     })
    #     return analysis

    canopy_bbox = features['canopy']['bbox']
    canopy_width = canopy_bbox[2] - canopy_bbox[0]
    canopy_height = canopy_bbox[3] - canopy_bbox[1]

    thresholds = indicator['tree_proportions']['relative_to_image']

    # For Width
    if canopy_width / image_width < thresholds["small_threshold"]:
        analysis.append({
            "trait": "ביטחון עצמי - רוחב",
            "description": thresholds["interpretation"]["small"],
            "indicator": "חופה צרה מאוד"
        })
    elif canopy_width / image_width > thresholds["large_threshold"]:
        analysis.append({
            "trait": "ביטחון עצמי - רוחב",
            "description": thresholds["interpretation"]["large"],
            "indicator": "חופה רחבה מאוד"
        })

    # For Height
    if canopy_height / image_height < thresholds["small_threshold"]:
        analysis.append({
            "trait": "ביטחון עצמי - גובה",
            "description": thresholds["interpretation"]["small"],
            "indicator": "חופה נמוכה מאוד"
        })
    elif canopy_height / image_height > thresholds["large_threshold"]:
        analysis.append({
            "trait": "ביטחון עצמי - גובה",
            "description": thresholds["interpretation"]["large"],
            "indicator": "חופה גבוהה מאוד"
        })

    return analysis


def generate_personality_output(features, image_width, image_height, indicator):
    """
    Generate a user-friendly personality analysis based on tree features.

    Args:
        features (dict): Parsed tree features.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.
        indicator (dict): Indicator thresholds and interpretations.

    Returns:
        str: Personality analysis paragraph.
    """
    # Step 1: Analyze tree proportions
    proportion_analysis = analyze_tree_proportions(features, image_width, image_height, indicator)

    # Step 2: Analyze feature proportions
    feature_analysis = analyze_feature_proportions(features, indicator)

    # Step 3: Analyze tree location
    location_analysis = analyze_tree_location(features, image_width, image_height, indicator)

    # Step 4: Analyze shapes
    shape_analysis = analyze_tree_shapes(features, indicator)

    #individual_analysis = analyze_tree_size_individual(features, image_width, image_height, indicator)

    # Combine all insights
    all_analysis = proportion_analysis + feature_analysis + location_analysis + shape_analysis

    # Format the results as a readable output
    return "\n\n".join(
        [f"{item['trait']}: {item['description']}" for item in all_analysis]
    )
