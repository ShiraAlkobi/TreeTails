def parse_yolo_output(yolo_output, image_width, image_height, indicator):
    features = {}

    for detection in yolo_output:
        label = detection["label"]
        bbox = detection["bbox"]
        confidence = detection["confidence"]

        # Calculate center
        center_x = (bbox[0] + bbox[2]) / 2 / image_width
        center_y = (bbox[1] + bbox[3]) / 2 / image_height

        # Calculate width and height of bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Normalize width and height by image dimensions
        norm_w = width / image_width
        norm_h = height / image_height

        if label == "canopy":
            # Determine canopy shape based on width-to-height ratio
            aspect_ratio = norm_w / norm_h
            if aspect_ratio > 0.8:  # Canopy is rounded
                shape = "rounded"
            else:  # Canopy is pointed
                shape = "pointed"

            # Determine canopy size
            size_thresholds = indicator["canopy"]["size"]
            if norm_w < size_thresholds["small_threshold"]:
                size = "small"
            elif norm_w > size_thresholds["large_threshold"]:
                size = "large"
            else:
                size = "none"


        elif label == "trunk":
            # Determine trunk shape based on curvature
            # Special shape handling for the trunk
            # Check for a center shift: If the trunk's center moves significantly from top to bottom
            center_shift = abs((bbox[0] + bbox[2]) / 2 - (image_width / 2)) / image_width

            # Check for significant width variation (if width changes a lot from top to bottom)
            width_variation = width / height  # Larger values suggest a more uneven shape

            if center_shift > 0.1 and width_variation > 0.05:
                shape = "curved"
            else:
                shape = "straight"

            size = "none"


        elif label == "root":
            shape = norm_h
            size = "none"


        features[label] = {
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "confidence": confidence,
            "shape": shape,
            "size": size
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
    analysis = [[], []]

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
        analysis[0].append({
            "trait": "ביטחון עצמי",
            "description": thresholds["interpretation"]["small"],
            "indicator": "הפרופורציה בין העץ המצויר לשאר הדף קטנה יחסית"
        })
        analysis[1].append({
            "לא מחפש להיות במרכז העניינים, עם רגליים על הקרקע"
        })
    elif tree_area_ratio > thresholds["large_threshold"]:
        analysis[0].append({
            "trait": "ביטחון עצמי",
            "description": thresholds["interpretation"]["large"],
            "indicator": "הפרופורציה בין העץ המצויר לשאר הדף גדולה יחסית"
        })
        analysis[1].append({
            "בעל נוכחות חזקהף, בלי פחד להיות בפרונט"
        })

    # Trunk-to-canopy ratio analysis
    trunk_height = trunk_bbox[3] - trunk_bbox[1]
    canopy_height = canopy_bbox[3] - canopy_bbox[1]
    trunk_canopy_ratio = trunk_height / canopy_height

    trunk_thresholds = indicator['tree_proportions']['features_relative_to_each_other']['trunk_canopy_ratio']
    if trunk_canopy_ratio < trunk_thresholds["thin_trunk_threshold"]:
        analysis[0].append({
            "trait": "אופי גמיש",
            "description": trunk_thresholds["interpretation"]["thin_trunk"],
            "indicator": "היחס בין הגזע לחופה מעיד על גמישות"
        })
        analysis[1].append({
            "לוקח דברים בקלילות וזורם עם שינויים"
        })
    elif trunk_canopy_ratio > trunk_thresholds["thick_trunk_threshold"]:
        analysis[0].append({
            "trait": "יציבות וביטחון",
            "description": trunk_thresholds["interpretation"]["thick_trunk"],
            "indicator": "היחס בין הגזע לחופה מעיד על יציבות"
        })
        analysis[1].append({
            "יציב, מחושב, ויודע מה הכיוון שלו"
        })
    # Root presence analysis
    root_thresholds = indicator['tree_proportions']['features_relative_to_each_other']['root_presence']
    if "root" not in features:
        analysis[0].append({
            "trait": "שלב חיפוש",
            "description": root_thresholds["interpretation"]["absent"],
            "indicator": "השורשים חסרים בציור"
        })
        analysis[1].append({
            "מוכן לשינויים, ומעדיף לא להיתקע במקום אחד"
        })
    else:
        analysis[0].append({
            "trait": "חיבור לערכים",
            "description": root_thresholds["interpretation"]["present"],
            "indicator": "השורשים נוכחים בציור"
        })
        analysis[1].append({
            "מחובר לערכים עם קרקע יציבה"
        })

    return analysis



# def analyze_feature_proportions(features, indicator):
#     """
#     Analyze the proportions of tree features relative to each other.
#
#     Args:
#         features (dict): Parsed tree features.
#         indicator (dict): Indicator thresholds and interpretations.
#
#     Returns:
#         list: Insights related to the relationship between features (e.g., trunk vs. canopy).
#     """
#     analysis = []
#
#     if "trunk" in features and "canopy" in features:
#         trunk_bbox = features['trunk']['bbox']
#         canopy_bbox = features['canopy']['bbox']
#
#         trunk_width = trunk_bbox[2] - trunk_bbox[0]
#         canopy_width = canopy_bbox[2] - canopy_bbox[0]
#
#         # # Check trunk size relative to canopy
#         # thresholds = indicator['tree_proportions']['features_relative_to_each_other']['trunk_canopy_ratio']
#         # if trunk_width < canopy_width * thresholds["thin_trunk_threshold"]:
#         #     analysis.append({
#         #         "trait": "כוח פנימי",
#         #         "description": thresholds["interpretation"]["thin_trunk"],
#         #         "indicator": "הגזע דק יחסית לחופה"
#         #     })
#         # elif trunk_width > canopy_width * thresholds["thick_trunk_threshold"]:
#         #     analysis.append({
#         #         "trait": "מיקוד יציב",
#         #         "description": thresholds["interpretation"]["thick_trunk"],
#         #         "indicator": "הגזע רחב יחסית לחופה"
#         #     })
#
#     # Root presence analysis
#     root_thresholds = indicator['tree_proportions']['features_relative_to_each_other']['root_presence']
#     if "root" in features:
#         analysis.append({
#             "trait": "נוכחות שורשים",
#             "description": root_thresholds["interpretation"]["present"],
#             "indicator": "נוכחות שורשים בציור"
#         })
#     else:
#         analysis.append({
#             "trait": "חסר שורשים",
#             "description": root_thresholds["interpretation"]["absent"],
#             "indicator": "חסרים שורשים בציור"
#         })
#
#     return analysis



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
    analysis = [[], []]

    # if "canopy" not in features:
    #     analysis.append({
    #         "trait": "חסר חופה",
    #         "description": "לא ניתן לנתח את מיקום העץ ללא חופה.",
    #         "indicator": "missing_canopy"
    #     })
    #     return analysis

    canopy_center_x = features['canopy']['center_x']
    canopy_center_y = features['canopy']['center_y']

    horizontal_thresholds = indicator['tree_location']['horizontal_position']
    vertical_thresholds = indicator['tree_location']['vertical_position']

    # Analyze horizontal position
    if canopy_center_x < horizontal_thresholds["left_threshold"]:
        analysis[0].append({
            "trait": "התמקדות בעבר",
            "description": horizontal_thresholds["interpretation"]["left"],
            "indicator": f"canopy_center_x < {horizontal_thresholds['left_threshold']}"
        })
        analysis[1].append({
            "טיפוס נוסטלגי עם חיבור לעברו ולערכיו"
        })
    elif canopy_center_x > horizontal_thresholds["right_threshold"]:
        analysis[0].append({
            "trait": "התמקדות בעתיד",
            "description": horizontal_thresholds["interpretation"]["right"],
            "indicator": f"canopy_center_x > {horizontal_thresholds['right_threshold']}"
        })
        analysis[1].append({
            "בעל שאיפות גדולות ומטרה ברורה קדימה"
        })
    elif (horizontal_thresholds["center_1_threshold"] <= canopy_center_x and
                horizontal_thresholds["center_2_threshold"] >= canopy_center_x):
        analysis[0].append({
            "trait": "תפיסה מאוזנת",
            "description": horizontal_thresholds["interpretation"]["center"],
            "indicator": f"{horizontal_thresholds['center_1_threshold']} <= canopy_center_x <= {horizontal_thresholds['center_2_threshold']}"
        })
        analysis[1].append({
            "מאוזן, יציב ויודע לשמור על עצמו גם בתקופות מסעירות"
        })

    # Analyze vertical position
    if canopy_center_y < vertical_thresholds["low_threshold"]:
        analysis[0].append({
            "trait": "מחובר לקרקע",
            "description": vertical_thresholds["interpretation"]["low"],
            "indicator": f"canopy_center_y < {vertical_thresholds['low_threshold']}"
        })
        analysis[1].append({
            "מחפש יציבות"
        })
    elif canopy_center_y > vertical_thresholds["high_threshold"]:
        analysis[0].append({
            "trait": "שאיפות גבוהות",
            "description": vertical_thresholds["interpretation"]["high"],
            "indicator": f"canopy_center_y > {vertical_thresholds['high_threshold']}"
        })
        analysis[1].append({
            "לא מפחד לשאוף רחוק עם מטרות גדולות"
        })
    elif (vertical_thresholds["center_1_threshold"] <= canopy_center_y and
                vertical_thresholds["center_2_threshold"] >= canopy_center_y):
        analysis[0].append({
            "trait": "יציבות רגשית",
            "description": vertical_thresholds["interpretation"]["center"],
            "indicator": f"{vertical_thresholds['center_1_threshold']} <= canopy_center_y <= {vertical_thresholds['center_2_threshold']}"
        })
        analysis[1].append({
            "מאוזן, מחובר לעצמו ויודע לשמור על פוקוס"
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
    analysis = [[], []]

    # Analyze canopy shape
    if "canopy" in features and "shape" in features['canopy']:
        shape = features['canopy']['shape']
        if shape in indicator['canopy']['shape']:
            analysis[0].append({
                "trait": "צורת כתר",
                "description": indicator['canopy']['shape'][shape],
                "indicator": f"canopy_shape = {shape}"
            })

            if shape == "rounded":
                analysis[1].append({
                    "מחפש הרמוניה ואיזון בחיים"
                })
            elif shape == "pointed":
                analysis[1].append({
                    "ממוקד וחד במטרותיו ובדרך שלו"
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
            analysis[0].append({
                "trait": "צורת גזע",
                "description": indicator['trunk']['shape'][shape],
                "indicator": f"trunk_shape = {shape}"
            })

            if shape == "straight":
                analysis[1].append({
                    "יציב ובטוח, פועל בצורה ישירה ולא מתפתלת"
                })
            elif shape == "curved":
                analysis[1].append({
                    "גמיש ומסתגל בקלות בכל סביבה"
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
    if "root" in features and "shape" in features['root']:
        root_depth = features['root']['shape']
        thresholds = indicator['roots']['shape']
        if root_depth < thresholds["shallow_threshold"]:
            analysis[0].append({
                "trait": "שורשים רדודים",
                "description": thresholds["interpretation"]["shallow"],
                "indicator": f"root_depth < {thresholds['shallow_threshold']}"
            })
            analysis[1].append({
                "זורם עם החיים ולא נקשר חזק למקום מסוים"
            })
        elif root_depth > thresholds["deep_threshold"]:
            analysis[0].append({
                "trait": "שורשים עמוקים",
                "description": thresholds["interpretation"]["deep"],
                "indicator": f"root_depth > {thresholds['deep_threshold']}"
            })
            analysis[1].append({
                "מחובר לערכים ועומד על יסודות יציבים"
            })
        # else:
        #     analysis.append({
        #         "trait": "שורשים בעומק בינוני",
        #         "description": "העומק של השורשים מתאים לאמצע הסקאלה.",
        #         "indicator": f"shallow_threshold <= root_depth <= deep_threshold"
        #     })
    # else:
    #     analysis.append({
    #         "trait": "חסרי שורשים",
    #         "description": "לא ניתן לנתח את עומק השורשים כי הם חסרים.",
    #         "indicator": "missing_root_depth"
    #     })

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

    # Step 2: Analyze tree location
    location_analysis = analyze_tree_location(features, image_width, image_height, indicator)

    # Step 3: Analyze shapes
    shape_analysis = analyze_tree_shapes(features, indicator)

    #individual_analysis = analyze_tree_size_individual(features, image_width, image_height, indicator)

    # Combine all insights
    fullAnalysis = proportion_analysis[0] + location_analysis[0] + shape_analysis[0]
    shortAnalysis = proportion_analysis[1] + location_analysis[1] + shape_analysis[1]

    # Format the results as a readable output
    fullAnalysisString = "\n\n".join(
        [f"{item['trait']}: {item['description']}" for item in fullAnalysis]
    )

    return fullAnalysisString