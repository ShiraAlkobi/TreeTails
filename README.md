# 🌳 TreeTails

**AI-Powered Personality Analysis from Tree Drawings**  
[GitHub Repository](https://github.com/shirka365/TreeTails)

---

## 🧠 Overview

**TreeTails** is an AI-based tool that analyzes hand-drawn trees to uncover personality traits. Using **state-of-the-art object detection (YOLOv8)** and psychological research, the system identifies components in tree drawings—**roots**, **trunk**, and **canopy**—to provide insight into the emotional and cognitive patterns of the drawer.

---

## 🎯 Objectives

- Combine **computer vision** with **projective psychology**.
- Deliver fast, visual personality analysis using **deep learning**.
- Create an engaging, web-based tool for both personal use and clinical potential.

---

## 📊 Dataset & Preprocessing

- ~1000 hand-drawn tree images with bounding box annotations:
  - `root`, `trunk`, `canopy`
- Labeled data converted from Excel → CSV → TFRecord & YOLO format.
- Data augmented to improve detection of underrepresented features (especially **roots**).

---

## 🤖 Model Architecture

### 🔍 YOLOv8 – Object Detection Core

The system uses **YOLOv8 (You Only Look Once)** by Ultralytics to detect tree components in user-uploaded drawings.

**Why YOLOv8?**
- Optimized for real-time object detection.
- Handles artistic variation better than SSD and custom CNNs.
- High accuracy and fast inference speed.

**Key Steps:**
- Converted data to YOLO TXT format:  
  `<class_id> <x_center> <y_center> <width> <height>`
- Trained with weighted loss emphasis on the **root** class.
- Used **data augmentation** (contrast adjustment, cropping, noise) to enhance performance.
- Boosted dataset by duplicating root-labeled images to resolve class imbalance.

---

## 📈 Feature Extraction & Analysis

After YOLOv8 detects the tree parts, custom logic extracts key visual metrics:
- Relative size ratios between parts
- Symmetry
- Line thickness and sharpness
- Root presence and depth

These metrics are stored in a structured `indicator.json` file and mapped to personality insights based on psychological studies.

---

## 🌐 Web Interface

Built with:
- **React Native** frontend
- **Flask** backend

Features:
- Upload hand-drawn tree
- YOLOv8-powered real-time analysis
- Receive visual and textual feedback
- Share results or vote on others' drawings

---

## 🔮 Future Enhancements

- Extend analysis to other drawing types (e.g. faces, handwriting)
- Visualize results on the drawing itself
- Enable longitudinal personality tracking
- Integrate into remote psychological tools and research

---

## 💡 Summary

TreeTails demonstrates how **modern object detection (YOLOv8)** can be applied beyond traditional use cases—bridging art, psychology, and machine learning. By analyzing the symbolic structure of a tree drawing, TreeTails offers a fresh, data-driven look into human personality.

